import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type Database from 'better-sqlite3';
import { openMemoryDatabase } from '../src/db.js';
import { search, searchBM25, buildFtsQuery } from '../src/search.js';
import type { SearchContext } from '../src/models.js';
import { FakeReranker } from './helpers/fixtures.js';

function seed(db: Database.Database): void {
  db.prepare(
    `INSERT INTO conv_sessions (id, project, jsonl_path, started_at) VALUES
     ('s1', '/a', '/t/s1', '2026-01-01'), ('s2', '/b', '/t/s2', '2026-01-01')`,
  ).run();
  const ins = db.prepare(
    `INSERT INTO conv_chunks (id, session_id, idx, user_content, assistant_content, hash, timestamp)
     VALUES (?, ?, ?, ?, ?, ?, ?)`,
  );
  ins.run('strong', 's1', 0, 'cors cors cors error', 'cors fix cors', 'h1', '2026-01-01T00:00:00Z');
  ins.run('weak', 's2', 0, 'networking error', 'mentions cors once', 'h2', '2026-01-02T00:00:00Z');
}

describe('one score direction (A8)', () => {
  let db: Database.Database;
  beforeEach(() => {
    db = openMemoryDatabase().db;
    seed(db);
  });
  afterEach(() => db.close());

  it('BM25 scores are higher-is-better', () => {
    const r = searchBM25(db, buildFtsQuery('cors'), 10);
    expect(r.map((x) => x.chunkId)).toEqual(['strong', 'weak']);
    expect(r[0].score).toBeGreaterThan(r[1].score);
  });

  it('an affinity that matches nothing keeps keyword order (vectors and reranker off)', async () => {
    const plain = await search(db, { query: 'cors', limit: 10 });
    const unrelated = await search(db, {
      query: 'cors',
      limit: 10,
      currentProject: '/nowhere',
      currentSession: 'nobody',
    });
    expect(plain.map((r) => r.chunkId)).toEqual(['strong', 'weak']);
    expect(unrelated.map((r) => r.chunkId)).toEqual(['strong', 'weak']);
    expect(unrelated.every((r) => r.score > 0)).toBe(true);
  });

  it('a session boost promotes its session even when the reranker returns negative logits', async () => {
    const ctx: SearchContext = {
      embedderText: null,
      embedderCode: null,
      reranker: new FakeReranker(), // scores are word hits minus 5: negative here
      vecTextEnabled: false,
      vecCodeEnabled: false,
      autoFuzzyThreshold: 3,
    };
    const r = await search(db, { query: 'cors', limit: 10, currentSession: 'nobody' }, ctx);
    expect(r.map((x) => x.chunkId)).toEqual(['strong', 'weak']);
    expect(r.every((x) => x.score > 0)).toBe(true);
    const boosted = await search(db, { query: 'cors', limit: 10, currentSession: 's2' }, ctx);
    expect(boosted[0].chunkId).toBe('weak');
  });

  it('a fuzzy-only hit is fused as its own channel with a positive score', async () => {
    const r = await search(db, { query: 'netw', limit: 10 }); // prefix of "networking"
    expect(r.map((x) => x.chunkId)).toEqual(['weak']);
    expect(r[0].matchType).toBe('fuzzy');
    expect(r[0].score).toBeGreaterThan(0);
  });
});
