import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type Database from 'better-sqlite3';
import { openMemoryDatabase } from '../src/db.js';
import { search, channelDepth, MAX_CANDIDATE_DEPTH, RERANK_POOL } from '../src/search.js';
import { registerSearchTools } from '../src/tools/search.js';
import type { SearchContext } from '../src/models.js';
import { FakeReranker, captureTools } from './helpers/fixtures.js';

describe('channelDepth', () => {
  it('keeps limit × 2 unless a larger minimum is set', () => {
    expect(channelDepth(10)).toBe(20);
    expect(channelDepth(10, 50)).toBe(50);
    expect(channelDepth(10, 5)).toBe(20);
    expect(channelDepth(10, 1000)).toBe(MAX_CANDIDATE_DEPTH);
  });

  it('ignores a depth that is not a number', () => {
    expect(channelDepth(10, Number.NaN)).toBe(20);
    expect(channelDepth(10, '50' as unknown as number)).toBe(20);
  });
});

describe('candidate depth in search', () => {
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
    for (let i = 0; i < 60; i++) ins.run(`c${i}`, i, `depth test item ${i}`, `h${i}`);
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

  // The reranker sees min(RERANK_POOL, fused candidates), so its pool shows the keyword depth.
  it('pulls the configured depth per channel; the rerank pool stays capped', async () => {
    const pools: number[] = [];
    for (const candidateDepth of [undefined, 15, 50]) {
      reranker.poolSizes = [];
      const r = await search(db, { query: 'depth test', limit: 5, candidateDepth }, ctx);
      expect(r).toHaveLength(5);
      pools.push(...reranker.poolSizes);
    }
    expect(pools).toEqual([10, 15, RERANK_POOL]);
  });

  it('m9k_search passes the configured depth through', async () => {
    const tools = captureTools(registerSearchTools, {
      db,
      cfg: { candidateDepth: 15 },
      searchContext: ctx,
    });
    await tools['m9k_search']({ query: 'depth test', limit: 5 });
    expect(reranker.poolSizes).toEqual([15]);
  });
});
