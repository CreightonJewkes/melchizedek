import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import { openDatabase, openMemoryDatabase, publishVectors } from '../src/db.js';
import { embedChunks } from '../src/indexer.js';
import {
  FakeEmbedder,
  seedSyntheticCorpus,
  vectorCounts,
  tempDbPath,
  sha,
} from './helpers/fixtures.js';

/** Every mapped vector must equal the embedding of its chunk's current text. */
async function staleMapped(db: Database.Database, suffix = '_text'): Promise<string[]> {
  const ref = new FakeEmbedder();
  const rows = db
    .prepare(
      `SELECT m.chunk_id, v.embedding, c.user_content || ' ' || c.assistant_content AS content
       FROM conv_vec_map${suffix} m JOIN conv_vec${suffix} v ON v.rowid = m.vec_rowid
       JOIN conv_chunks c ON c.id = m.chunk_id`,
    )
    .all() as Array<{ chunk_id: string; embedding: Buffer; content: string }>;
  const stale: string[] = [];
  for (const r of rows) {
    const stored = new Float32Array(
      r.embedding.buffer,
      r.embedding.byteOffset,
      r.embedding.length / 4,
    );
    const want = await ref.embed(r.content);
    if (want.some((x, i) => Math.abs(x - stored[i]) > 1e-6)) stale.push(r.chunk_id);
  }
  return stale;
}

function rewriteChunk(db: Database.Database, id: string, text: string): void {
  db.prepare('UPDATE conv_chunks SET user_content = ?, hash = ? WHERE id = ?').run(
    text,
    sha(text),
    id,
  );
}

describe('safe vector publication (L3)', () => {
  let db: Database.Database;

  beforeEach(() => {
    db = openMemoryDatabase().db;
    seedSyntheticCorpus(db);
  });
  afterEach(() => db.close());

  it('discards a vector whose chunk text changed during inference, then embeds the new text', async () => {
    const e = new FakeEmbedder();
    let fired = false;
    e.onBatch = () => {
      if (fired) return;
      fired = true;
      rewriteChunk(db, 'sess-live:0', 'Completely different text about kubernetes ingress rules');
    };
    await embedChunks(db, e, '_text', 50);
    expect(await staleMapped(db)).toEqual([]);
    const mapped = db
      .prepare("SELECT 1 AS x FROM conv_vec_map_text WHERE chunk_id = 'sess-live:0'")
      .get();
    expect(mapped).toEqual({ x: 1 });
    const c = vectorCounts(db, '_text');
    expect(c.vec).toBe(c.map);
  });

  it('a chunk deleted during inference publishes nothing and raises nothing', async () => {
    const e = new FakeEmbedder();
    e.onBatch = () => {
      db.prepare("DELETE FROM conv_chunks WHERE id = 'sess-live:2'").run();
    };
    await expect(embedChunks(db, e, '_text', 50)).resolves.toBeGreaterThan(0);
    const c = vectorCounts(db, '_text');
    expect(c.vec).toBe(c.map);
    expect(
      db
        .prepare("SELECT COUNT(*) AS n FROM conv_vec_map_text WHERE chunk_id = 'sess-live:2'")
        .get(),
    ).toEqual({ n: 0 });
  });

  it('publishing the same chunk twice leaves one vector and one map row', async () => {
    const row = db.prepare("SELECT hash FROM conv_chunks WHERE id = 'sess-live:0'").get() as {
      hash: string;
    };
    const emb = await new FakeEmbedder().embed('x');
    const item = { chunkId: 'sess-live:0', hash: row.hash, embedding: emb };
    expect(publishVectors(db, [item])).toBe(1);
    expect(publishVectors(db, [item])).toBe(0);
    expect(vectorCounts(db, '_text')).toEqual({ vec: 1, map: 1 });
  });

  it('counts only committed vectors as progress', async () => {
    const e = new FakeEmbedder();
    e.onBatch = (texts) => {
      if (texts.length > 1)
        rewriteChunk(db, 'sess-copy:0', 'changed while embedding, long enough text');
    };
    const progress: number[] = [];
    const done = await embedChunks(db, e, '_text', 50, (n) => progress.push(n));
    expect(done).toBe(vectorCounts(db, '_text').map);
  });
});

describe('two workers publishing at once (L3)', () => {
  let cleanup = (): void => {};
  afterEach(() => cleanup());

  it('every eligible chunk ends with exactly one current vector', async () => {
    const tmp = tempDbPath();
    cleanup = tmp.cleanup;
    const a = openDatabase(tmp.dbPath).db;
    seedSyntheticCorpus(a);
    const b = openDatabase(tmp.dbPath).db;

    await Promise.all([
      embedChunks(a, new FakeEmbedder('fake-model', 384, 5), '_text', 2),
      embedChunks(b, new FakeEmbedder('fake-model', 384, 3), '_text', 2),
    ]);

    const c = vectorCounts(a, '_text');
    expect(c.vec).toBe(c.map);
    const eligible = a
      .prepare(
        `SELECT COUNT(*) AS n FROM conv_chunks WHERE deleted_at IS NULL
         AND LENGTH(user_content) + LENGTH(assistant_content) >= 50`,
      )
      .get() as { n: number };
    expect(c.map).toBe(eligible.n);
    expect(await staleMapped(a)).toEqual([]);
    a.close();
    b.close();
  });
});
