import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type Database from 'better-sqlite3';
import { openMemoryDatabase } from '../src/db.js';
import { search, RERANK_POOL } from '../src/search.js';
import type { SearchContext } from '../src/models.js';
import { FakeReranker } from './helpers/fixtures.js';

describe('rerank pool (A10)', () => {
  let db: Database.Database;
  let reranker: FakeReranker;
  let ctx: SearchContext;

  beforeEach(() => {
    db = openMemoryDatabase().db;
    db.prepare(
      "INSERT INTO conv_sessions (id, project, jsonl_path, started_at) VALUES ('s', '/p', '/t/s', '2026-01-01')",
    ).run();
    const ins = db.prepare(
      `INSERT INTO conv_chunks (id, session_id, idx, user_content, assistant_content, hash, timestamp)
       VALUES (?, 's', ?, ?, '', ?, '2026-01-01T00:00:00Z')`,
    );
    // 120 keyword matches; later ones mention "budget" more, so the reranker prefers them.
    for (let i = 0; i < 120; i++) {
      ins.run(`c${i}`, i, `pool test ${'budget '.repeat(i % 7)}item ${i}`, `h${i}`);
    }
    reranker = new FakeReranker();
    ctx = {
      embedderText: null,
      embedderCode: null,
      reranker,
      vecTextEnabled: false,
      vecCodeEnabled: false,
      autoFuzzyThreshold: 3,
    };
  });
  afterEach(() => db.close());

  it('scores at most RERANK_POOL candidates, whatever the output limit', async () => {
    for (const limit of [5, 10, 50]) {
      reranker.poolSizes = [];
      const r = await search(db, { query: 'pool budget', limit }, ctx);
      expect(reranker.poolSizes).toEqual([Math.min(RERANK_POOL, 2 * limit)]);
      expect(r).toHaveLength(limit);
    }
  });

  it('keeps fused candidates beyond the pool, after the reranked ones, for large limits', async () => {
    const r = await search(db, { query: 'pool budget', limit: 50 }, ctx);
    expect(new Set(r.map((x) => x.chunkId)).size).toBe(50);
    for (let i = 1; i < r.length; i++) expect(r[i - 1].score).toBeGreaterThan(r[i].score);
  });
});
