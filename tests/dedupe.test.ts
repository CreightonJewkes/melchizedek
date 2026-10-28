import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type Database from 'better-sqlite3';
import { openMemoryDatabase } from '../src/db.js';
import { search } from '../src/search.js';

describe('duplicate exchanges and the per-session cap (A6)', () => {
  let db: Database.Database;
  beforeEach(() => {
    db = openMemoryDatabase().db;
    db.prepare(
      `INSERT INTO conv_sessions (id, project, jsonl_path, started_at) VALUES
       ('orig', '/p', '/t/orig', '2026-01-01'), ('fork', '/p', '/t/fork', '2026-01-05'),
       ('busy', '/p', '/t/busy', '2026-01-02'), ('other', '/p', '/t/other', '2026-01-03')`,
    ).run();
    const ins = db.prepare(
      `INSERT INTO conv_chunks (id, session_id, idx, user_content, assistant_content, hash, timestamp)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
    );
    // The same exchange copied into a forked session (newer), plus distinct hits.
    ins.run(
      'orig:0',
      'orig',
      0,
      'kafka consumer lag kafka',
      'raise partitions',
      'same',
      '2026-01-01T00:00:00Z',
    );
    ins.run(
      'fork:0',
      'fork',
      0,
      'kafka consumer lag kafka',
      'raise partitions',
      'same',
      '2026-01-05T00:00:00Z',
    );
    for (let i = 0; i < 4; i++) {
      ins.run(
        `busy:${i}`,
        'busy',
        i,
        `kafka kafka kafka note ${i}`,
        '',
        `b${i}`,
        '2026-01-02T00:00:00Z',
      );
    }
    ins.run('other:0', 'other', 0, 'kafka elsewhere', '', 'o0', '2026-01-03T00:00:00Z');
  });
  afterEach(() => db.close());

  it('collapses identical copies to the best-ranked one and lists the other sessions', async () => {
    const r = await search(db, { query: 'kafka consumer lag', limit: 10 });
    const copies = r.filter((x) => x.chunkId === 'orig:0' || x.chunkId === 'fork:0');
    expect(copies).toHaveLength(1);
    expect(copies[0].alsoIn).toEqual([copies[0].sessionId === 'orig' ? 'fork' : 'orig']);
    expect(r[0]).toBe(copies[0]); // the best-ranked copy keeps the top slot
  });

  it('the session cap is off by default and refills from lower ranks when set', async () => {
    const all = await search(db, { query: 'kafka', limit: 4 });
    expect(all.filter((x) => x.sessionId === 'busy').length).toBeGreaterThan(2);

    const capped = await search(db, { query: 'kafka', limit: 4, maxPerSession: 2 });
    expect(capped).toHaveLength(4);
    expect(capped.filter((x) => x.sessionId === 'busy')).toHaveLength(2);
    expect(capped.some((x) => x.sessionId === 'other')).toBe(true);
  });
});
