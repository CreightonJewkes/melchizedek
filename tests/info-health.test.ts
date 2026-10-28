import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type Database from 'better-sqlite3';
import {
  openMemoryDatabase,
  setMeta,
  getChunksWithoutEmbeddings,
  publishVectors,
} from '../src/db.js';
import { getConfig } from '../src/config.js';
import { seedEmbeddingMeta } from '../src/migration.js';
import { registerMemoryTools } from '../src/tools/memory.js';
import type { ToolContext } from '../src/tools/context.js';
import { captureTools, FakeEmbedder, seedSyntheticCorpus } from './helpers/fixtures.js';

describe('m9k_info reports index health (C3)', () => {
  let db: Database.Database;
  let ctx: ToolContext;
  let lastError: { message: string; at: string } | null;
  const embedder = new FakeEmbedder('fake-model');

  const info = async () =>
    JSON.parse((await captureTools(registerMemoryTools, ctx)['m9k_info']({})).content[0].text) as {
      status: string;
      health: {
        vectors: Array<{
          table: string;
          vectors: number;
          mapped: number;
          orphanVectors: number;
          danglingMaps: number;
        }>;
        pending: { unembedded: number; oldestSince: string | null; oldestMinutes: number | null };
        textVectors: string;
        loadedModel: { id: string } | null;
        lastJobError: { message: string } | null;
        keywordErrors: { syntax: number; busy: number; other: number };
        warnings: string[];
      };
    };

  beforeEach(async () => {
    db = openMemoryDatabase().db;
    seedSyntheticCorpus(db);
    seedEmbeddingMeta(db, embedder, '_text');
    const rows = getChunksWithoutEmbeddings(db, 100, '_text');
    const vecs = await embedder.embedBatch(rows.map((r) => r.content));
    publishVectors(
      db,
      rows.map((r, i) => ({ chunkId: r.id, hash: r.hash, embedding: vecs[i] })),
      '_text',
    );
    lastError = null;
    ctx = {
      db,
      cfg: getConfig({ dbPath: ':memory:' }),
      searchContext: {
        embedderText: embedder,
        embedderCode: null,
        reranker: null,
        vecTextEnabled: true,
        vecCodeEnabled: false,
        autoFuzzyThreshold: 3,
      },
      orchestrator: {
        getStatus: () => ({
          active: false,
          suffix: null,
          embedded: 0,
          total: 0,
          pid: null,
          rssMB: null,
          heapUsedMB: null,
        }),
        getLastError: () => lastError,
      } as unknown as ToolContext['orchestrator'],
      embeddingState: { get: () => false, set: () => undefined },
      version: 'test',
      mode: 'local',
    };
  });
  afterEach(() => db.close());

  it('a clean index is healthy, with every count present', async () => {
    const r = await info();
    expect(r.status).toContain('healthy');
    const text = r.health.vectors.find((v) => v.table === 'conv_vec_text')!;
    expect(text.orphanVectors).toBe(0);
    expect(text.vectors).toBe(text.mapped);
    expect(r.health.textVectors).toBe('active');
    expect(r.health.loadedModel?.id).toBe('fake-model');
    expect(r.health.pending.unembedded).toBe(0);
    expect(r.health.warnings).toEqual([]);
    expect(Object.keys(r.health.keywordErrors)).toEqual(['syntax', 'busy', 'other']);
  });

  it('orphans, a stale wait, a model mismatch and a failed job each degrade it, with reasons', async () => {
    const ins = db.prepare('INSERT INTO conv_vec_text (embedding) VALUES (?)');
    for (let i = 0; i < 50; i++) ins.run(Buffer.from(new Float32Array(384).fill(0.1).buffer));
    db.prepare(
      `INSERT INTO conv_chunks (id, session_id, idx, user_content, assistant_content, hash, timestamp, indexed_at)
       VALUES ('sess-live:9', 'sess-live', 9, ?, '', 'h-new', '2026-03-01T00:00:00Z', '2026-03-01T00:00:00Z')`,
    ).run('a newly indexed exchange that has been waiting for its vector for a long time');
    setMeta(db, 'embedding_recipe_text', 'some other recipe');
    lastError = { message: 'no progress for 300 s', at: '2026-03-01T00:00:00Z' };

    const r = await info();
    expect(r.status).toContain('degraded');
    expect(r.health.vectors.find((v) => v.table === 'conv_vec_text')!.orphanVectors).toBe(50);
    expect(r.health.pending.unembedded).toBe(1);
    expect(r.health.pending.oldestSince).toBe('2026-03-01T00:00:00Z');
    expect(r.health.textVectors).toMatch(/another model or recipe/);
    expect(r.health.lastJobError?.message).toMatch(/no progress/);
    expect(r.health.warnings.join(' | ')).toMatch(/orphans.*waited.*recipe.*no progress/);
  });
});
