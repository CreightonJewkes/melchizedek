import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type Database from 'better-sqlite3';
import {
  openMemoryDatabase,
  countEmbeddedChunks,
  createVecTables,
  deleteChunksWhere,
  setMeta,
  sweepOrphanVectors,
} from '../src/db.js';
import { embedChunks } from '../src/indexer.js';
import { searchVectorized } from '../src/search.js';
import { TransformersJsEmbedder } from '../src/embedder.js';
import { MODEL_REGISTRY } from '../src/constants.js';
import {
  seedEmbeddingMeta,
  detectModelChange,
  vectorIndexMatches,
  EmbeddingMigration,
} from '../src/migration.js';
import { baseSuffix, chunkWindows, hasSecondWindow, secondWindowSuffix } from '../src/windows.js';
import { FakeEmbedder, vectorCounts } from './helpers/fixtures.js';

/** A fake model whose recipe carries the windows switch, as the registry embedder's does. */
class WindowedFake extends FakeEmbedder {
  constructor(private readonly split: boolean) {
    super('fake-model');
  }
  splitWindows(): boolean {
    return this.split;
  }
  recipe(): string {
    return JSON.stringify({ fake: 1, ...(this.split ? { windows: true } : {}) });
  }
}

const PROMPT = 'alpha bravo charlie question about the deploy pipeline';
const REPLY = 'zulu yankee xray answer explaining the rollback steps in detail';

describe('split windows: units', () => {
  it('names the parallel pair and maps it back to its generation', () => {
    expect(secondWindowSuffix('_text')).toBe('_text_w2');
    expect(secondWindowSuffix('_text_next')).toBe('_text_w2_next');
    expect(baseSuffix('_text_w2_next')).toBe('_text');
    expect(baseSuffix('_text_w2')).toBe('_text');
    expect(baseSuffix('_code_next')).toBe('_code');
  });

  it('splits only when on, and leaves out an empty window', () => {
    expect(chunkWindows('a', 'b', false)).toEqual(['a b']);
    expect(chunkWindows('a', 'b', true)).toEqual(['a', 'b']);
    expect(chunkWindows('a', '  ', true)).toEqual(['a']);
  });

  it('keeps the stored recipe byte-identical when the option is off', () => {
    const spec = MODEL_REGISTRY['gte-small'];
    const off = new TransformersJsEmbedder(spec).recipe();
    expect(off).toBe(
      '{"hf":"Xenova/gte-small","pooling":"mean","query":"","document":"","input":2000,"dtype":"q8"}',
    );
    const on = new TransformersJsEmbedder({ ...spec, splitWindows: true });
    expect(JSON.parse(on.recipe()).windows).toBe(true);
    expect(on.splitWindows()).toBe(true);
  });
});

describe('split windows: storage, search and lifecycle', () => {
  let db: Database.Database;

  beforeEach(() => {
    db = openMemoryDatabase().db;
    db.prepare(
      "INSERT INTO conv_sessions (id, project, jsonl_path, started_at) VALUES ('s', '/p', '/t/s', '2026-01-01')",
    ).run();
    const ins = db.prepare(
      `INSERT INTO conv_chunks (id, session_id, idx, user_content, assistant_content, hash, timestamp)
       VALUES (?, 's', ?, ?, ?, ?, '2026-01-01T00:00:00Z')`,
    );
    ins.run('both', 0, PROMPT, REPLY, 'h0');
    ins.run('promptOnly', 1, 'a prompt with no reply that is long enough to embed here', '', 'h1');
    createVecTables(db, 384, '_text');
  });
  afterEach(() => db.close());

  it('embeds one vector per window and counts chunks once', async () => {
    const n = await embedChunks(db, new WindowedFake(true), '_text');
    expect(n).toBe(2);
    expect(countEmbeddedChunks(db, '_text')).toBe(2);
    expect(vectorCounts(db, '_text_w2')).toEqual({ vec: 1, map: 1 });
  });

  it('finds a chunk by its reply window, once', async () => {
    const emb = new WindowedFake(true);
    await embedChunks(db, emb, '_text');
    const q = await emb.embedQuery('zulu yankee xray rollback steps');
    const hits = searchVectorized(db, q, 5);
    expect(hits[0].chunkId).toBe('both');
    expect(new Set(hits.map((h) => h.chunkId)).size).toBe(hits.length);
    // The same query scored with a date scope takes the exact path and agrees.
    const scoped = searchVectorized(db, q, 5, 1.3, '_text', { since: '2025-01-01T00:00:00Z' });
    expect(scoped.map((h) => h.chunkId)).toEqual(hits.map((h) => h.chunkId));
  });

  it('the reply window makes the match: joined vectors rank it lower', async () => {
    const q = await new WindowedFake(true).embedQuery('zulu yankee xray rollback steps');
    await embedChunks(db, new WindowedFake(false), '_text');
    const joined = searchVectorized(db, q, 5)[0].score;
    db.exec('DELETE FROM conv_vec_map_text; DELETE FROM conv_vec_text;');
    await embedChunks(db, new WindowedFake(true), '_text');
    const split = searchVectorized(db, q, 5)[0].score;
    expect(split).toBeGreaterThan(joined);
  });

  it('switching windows on and off goes through the migration and its swap', async () => {
    const joined = new WindowedFake(false);
    seedEmbeddingMeta(db, joined, '_text');
    await embedChunks(db, joined, '_text');

    const split = new WindowedFake(true);
    expect(detectModelChange(db, split, '_text')).not.toBeNull();
    expect(vectorIndexMatches(db, split, '_text')).toBe(false);
    const on = await new EmbeddingMigration(db, split, { suffix: '_text' }).run();
    expect(on).toMatchObject({ swapped: true, embedded: 2 });
    expect(vectorIndexMatches(db, split, '_text')).toBe(true);
    expect(countEmbeddedChunks(db, '_text')).toBe(2);
    expect(vectorCounts(db, '_text_w2')).toEqual({ vec: 1, map: 1 });
    expect(hasSecondWindow(db, '_text_next')).toBe(false);

    const off = await new EmbeddingMigration(db, joined, { suffix: '_text' }).run();
    expect(off.swapped).toBe(true);
    expect(hasSecondWindow(db, '_text')).toBe(false);
    expect(countEmbeddedChunks(db, '_text')).toBe(2);
  });

  it('deleting a chunk removes its reply vector too', async () => {
    await embedChunks(db, new WindowedFake(true), '_text');
    deleteChunksWhere(db, 'id = ?', ['both']);
    expect(vectorCounts(db, '_text_w2')).toEqual({ vec: 0, map: 0 });
    expect(countEmbeddedChunks(db, '_text')).toBe(1);
  });

  it('the sweep leaves a migrating generation alone, reply pair included', () => {
    createVecTables(db, 384, '_text_w2_next');
    db.prepare('INSERT INTO conv_vec_text_w2_next (embedding) VALUES (?)').run(
      Buffer.from(new Float32Array(384).fill(0.01).buffer),
    );
    setMeta(db, 'migration_target_model_text', 'fake-model');
    const r = sweepOrphanVectors(db);
    expect(r.skipped).toContain('conv_vec_text_w2_next');
    expect(vectorCounts(db, '_text_w2_next').vec).toBe(1);
  });
});
