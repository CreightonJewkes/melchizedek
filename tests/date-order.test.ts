import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type Database from 'better-sqlite3';
import { openMemoryDatabase } from '../src/db.js';
import { search } from '../src/search.js';
import type { SearchContext } from '../src/models.js';
import { FakeReranker } from './helpers/fixtures.js';

describe('chronological order (A9)', () => {
  let db: Database.Database;
  let ins: Database.Statement;

  beforeEach(() => {
    db = openMemoryDatabase().db;
    db.prepare(
      "INSERT INTO conv_sessions (id, project, jsonl_path, started_at) VALUES ('s', '/p', '/t/s', '2026-01-01')",
    ).run();
    ins = db.prepare(
      `INSERT INTO conv_chunks (id, session_id, idx, user_content, assistant_content, hash, timestamp)
       VALUES (?, 's', ?, ?, '', ?, ?)`,
    );
    // The first mention is one word in a long exchange, so it ranks last on relevance;
    // 40 later exchanges repeat the term and fill the relevance pool many times over.
    ins.run(
      'first',
      0,
      `kafka ${'unrelated filler words '.repeat(60)}`,
      'h-first',
      '2026-01-01T09:00:00Z',
    );
    for (let i = 1; i <= 40; i++) {
      const day = String(1 + (i % 28)).padStart(2, '0');
      ins.run(
        `c${i}`,
        i,
        'kafka kafka kafka kafka',
        `h${i}`,
        `2026-02-${day}T10:${String(i).padStart(2, '0')}:00Z`,
      );
    }
  });
  afterEach(() => db.close());

  it('finds the first occurrence even when it ranks below the relevance cut', async () => {
    const ranked = await search(db, { query: 'kafka', limit: 5 });
    expect(ranked.map((r) => r.chunkId)).not.toContain('first');

    const asc = await search(db, { query: 'kafka', limit: 5, order: 'date_asc' });
    expect(asc[0].chunkId).toBe('first');
    expect(asc).toHaveLength(5);
    for (let i = 1; i < asc.length; i++) {
      expect(asc[i - 1].timestamp <= asc[i].timestamp).toBe(true);
    }
  });

  it('date_desc returns the latest matches, newest first', async () => {
    const desc = await search(db, { query: 'kafka', limit: 3, order: 'date_desc' });
    const all = db
      .prepare('SELECT id FROM conv_chunks ORDER BY timestamp DESC LIMIT 3')
      .all() as Array<{ id: string }>;
    expect(desc.map((r) => r.chunkId)).toEqual(all.map((r) => r.id));
  });

  it('applies the scope before choosing the earliest match', async () => {
    const r = await search(db, {
      query: 'kafka',
      limit: 5,
      order: 'date_asc',
      since: '2026-02-10T00:00:00Z',
    });
    expect(r).toHaveLength(5);
    expect(r.every((x) => x.timestamp >= '2026-02-10T00:00:00Z')).toBe(true);
    const earliest = db
      .prepare(
        "SELECT min(timestamp) AS t FROM conv_chunks WHERE timestamp >= '2026-02-10T00:00:00Z'",
      )
      .get() as { t: string };
    expect(r[0].timestamp).toBe(earliest.t);
  });

  it('takes exchanges with every term first, then fills from the relevance pool', async () => {
    ins.run('both', 50, 'kafka consumer lag', 'h-both', '2026-03-01T00:00:00Z');
    const r = await search(db, { query: 'kafka lag', limit: 4, order: 'date_asc' });
    expect(r).toHaveLength(4);
    expect(r.map((x) => x.chunkId)).toContain('both'); // the newest, but the only full match
    for (let i = 1; i < r.length; i++) {
      expect(r[i - 1].timestamp <= r[i].timestamp).toBe(true);
    }
  });

  it('skips the reranker: time order wins over relevance order', async () => {
    const reranker = new FakeReranker();
    const ctx: SearchContext = {
      embedderText: null,
      embedderCode: null,
      reranker,
      vecTextEnabled: false,
      vecCodeEnabled: false,
      autoFuzzyThreshold: 3,
    };
    const asc = await search(db, { query: 'kafka', limit: 5, order: 'date_asc' }, ctx);
    expect(reranker.poolSizes).toEqual([]);
    expect(asc[0].chunkId).toBe('first');

    await search(db, { query: 'kafka', limit: 5 }, ctx);
    expect(reranker.poolSizes).toHaveLength(1); // relevance order still reranks
  });
});
