import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import {
  openDatabase,
  closeDatabase,
  getStat,
  readStat,
  flushStats,
  incrementStat,
} from '../src/db.js';
import { search } from '../src/search.js';
import { tempDbPath } from './helpers/fixtures.js';

describe('search usage counters off the request path (A11)', () => {
  let tmp: ReturnType<typeof tempDbPath>;
  let db: Database.Database;
  let other: Database.Database | undefined;

  beforeEach(() => {
    tmp = tempDbPath('m9k-stats-');
    db = openDatabase(tmp.dbPath).db;
    db.prepare(
      "INSERT INTO conv_sessions (id, project, jsonl_path, started_at) VALUES ('s', '/p', '/t/s', '2026-01-01')",
    ).run();
    const ins = db.prepare(
      `INSERT INTO conv_chunks (id, session_id, idx, user_content, assistant_content, hash, timestamp)
       VALUES (?, 's', ?, ?, 'answer', ?, '2026-01-01T00:00:00Z')`,
    );
    for (let i = 0; i < 3; i++) ins.run(`c${i}`, i, `kafka note ${i}`, `h${i}`);
  });
  afterEach(() => {
    other?.close();
    other = undefined;
    if (db.open) db.close();
    tmp.cleanup();
  });

  it("does not wait on another writer's lock, and a flush under the lock is dropped quickly", async () => {
    other = new Database(tmp.dbPath);
    other.exec('BEGIN IMMEDIATE'); // a hook mid-transaction in another process
    const t0 = performance.now();
    const r = await search(db, { query: 'kafka', limit: 5 });
    expect(r.length).toBeGreaterThan(0);
    expect(performance.now() - t0).toBeLessThan(1000); // busy_timeout is 5,000 ms

    const t1 = performance.now();
    expect(flushStats(db)).toBe(false);
    expect(performance.now() - t1).toBeLessThan(1000);
    other.exec('COMMIT');

    expect(flushStats(db)).toBe(true); // nothing left: the dropped counts are gone
    expect(getStat(db, 'search_count')).toBeNull();
  });

  it('keeps counts in memory and writes them in one flush', async () => {
    await search(db, { query: 'kafka', limit: 5 });
    await search(db, { query: 'nothingmatcheshere', limit: 5 });
    expect(getStat(db, 'search_count')).toBeNull();

    expect(flushStats(db)).toBe(true);
    expect(getStat(db, 'search_count')).toBe('2');
    expect(getStat(db, 'hit_count')).toBe('1');
    expect(Number(getStat(db, 'tokens_served'))).toBeGreaterThan(0);
    expect(getStat(db, 'last_search_at')).not.toBeNull();
  });

  it('readers see queued counts before the flush (m9k_info, usage guide)', async () => {
    incrementStat(db, 'search_count', 5);
    await search(db, { query: 'kafka', limit: 5 });
    expect(getStat(db, 'search_count')).toBe('5');
    expect(readStat(db, 'search_count')).toBe('6');
    expect(readStat(db, 'last_search_at')).not.toBeNull();
    flushStats(db);
    expect(readStat(db, 'search_count')).toBe('6');
  });

  it('increments atomically when another connection writes the same counter', () => {
    other = new Database(tmp.dbPath);
    incrementStat(db, 'search_count', 3);
    incrementStat(other, 'search_count', 4);
    incrementStat(db, 'search_count');
    expect(getStat(db, 'search_count')).toBe('8');
  });

  it('closeDatabase flushes pending counts first', async () => {
    await search(db, { query: 'kafka', limit: 5 });
    closeDatabase(db);
    const reopened = new Database(tmp.dbPath, { readonly: true });
    expect(getStat(reopened, 'search_count')).toBe('1');
    reopened.close();
  });
});
