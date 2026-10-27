import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type Database from 'better-sqlite3';
import { openMemoryDatabase } from '../src/db.js';
import {
  normalizeDateBound,
  resolveTimeframe,
  resolveDateRange,
  DateInputError,
} from '../src/dates.js';
import { registerSearchTools } from '../src/tools/search.js';
import { registerSpecializedTools } from '../src/tools/specialized.js';
import { registerMemoryTools } from '../src/tools/memory.js';
import { captureTools, seedSyntheticCorpus } from './helpers/fixtures.js';

// Wednesday 2026-03-04 15:30 local time.
const NOW = new Date(2026, 2, 4, 15, 30);
const local = (...a: [number, number, number, number?, number?]): string =>
  new Date(a[0], a[1], a[2], a[3] ?? 0, a[4] ?? 0).toISOString();

describe('relative dates (A5)', () => {
  it('resolves relative points against the injected clock, in local time', () => {
    expect(normalizeDateBound('today', 'since', NOW)).toBe(local(2026, 2, 4));
    expect(normalizeDateBound('yesterday', 'since', NOW)).toBe(local(2026, 2, 3));
    expect(normalizeDateBound('3 days ago', 'since', NOW)).toBe(local(2026, 2, 1, 15, 30));
    expect(normalizeDateBound('2 weeks ago', 'since', NOW)).toBe(local(2026, 1, 18, 15, 30));
    expect(normalizeDateBound('a couple hours ago', 'since', NOW)).toBe(
      new Date(NOW.getTime() - 2 * 3_600_000).toISOString(),
    );
    expect(normalizeDateBound('a few  days ago', 'since', NOW)).toBe(local(2026, 2, 1, 15, 30));
    expect(normalizeDateBound('an hour ago', 'since', NOW)).toBe(
      new Date(NOW.getTime() - 3_600_000).toISOString(),
    );
    expect(normalizeDateBound('last month', 'since', NOW)).toBe(local(2026, 1, 4, 15, 30));
  });

  it('day arithmetic follows the local calendar across a DST change', () => {
    // 2026-03-08 is a US DST start; "1 day ago" from 03-09 noon is 03-08 noon local.
    const after = new Date(2026, 2, 9, 12, 0);
    expect(normalizeDateBound('1 day ago', 'since', after)).toBe(local(2026, 2, 8, 12, 0));
  });

  it('timeframes are local ranges with an exclusive until', () => {
    expect(resolveTimeframe('yesterday', NOW)).toEqual({
      since: local(2026, 2, 3),
      until: local(2026, 2, 4),
    });
    expect(resolveTimeframe('this week', NOW)).toEqual({ since: local(2026, 2, 1) }); // Sunday
    expect(resolveTimeframe('last week', NOW)).toEqual({
      since: local(2026, 1, 22),
      until: local(2026, 2, 1),
    });
    expect(resolveTimeframe('last month', NOW)).toEqual({
      since: local(2026, 1, 1),
      until: local(2026, 2, 1),
    });
    expect(resolveTimeframe('past 3 days', NOW)).toEqual({ since: local(2026, 2, 1, 15, 30) });
    expect(resolveTimeframe('3 hours ago', NOW)).toEqual({
      since: new Date(NOW.getTime() - 4 * 3_600_000).toISOString(),
      until: new Date(NOW.getTime() - 2 * 3_600_000).toISOString(),
    });
  });

  it('explicit since/until override the matching side of a timeframe', () => {
    expect(resolveDateRange({ timeframe: 'last week', until: '2026-02-25' }, NOW)).toEqual({
      since: local(2026, 1, 22),
      until: local(2026, 1, 25),
    });
  });

  it('rejects what it does not support instead of passing it through', () => {
    for (const bad of ['next week', 'soon', '2026-10-13 garbage', 'ages ago', '3 fortnights ago']) {
      expect(() => normalizeDateBound(bad, 'since', NOW)).toThrow(DateInputError);
    }
    expect(() => resolveTimeframe('whenever', NOW)).toThrow(DateInputError);
  });
});

describe('date handling in the tools (A5)', () => {
  let db: Database.Database;
  const clock = (): Date => new Date('2026-03-02T10:00:00.000Z');
  let tools: ReturnType<typeof captureTools>;
  const parse = (r: { content: Array<{ text: string }> }): Array<Record<string, string>> =>
    JSON.parse(r.content[0].text);

  beforeEach(() => {
    db = openMemoryDatabase().db;
    seedSyntheticCorpus(db);
    const ctx = {
      db,
      clock,
      cfg: {},
      searchContext: {
        embedderText: null,
        embedderCode: null,
        reranker: null,
        vecTextEnabled: false,
        vecCodeEnabled: false,
        autoFuzzyThreshold: 3,
      },
    };
    tools = {
      ...captureTools(registerSearchTools, ctx),
      ...captureTools(registerSpecializedTools, ctx),
      ...captureTools(registerMemoryTools, ctx),
    };
  });
  afterEach(() => db.close());

  it('m9k_search, m9k_similar_work and m9k_sessions accept timeframe and relative dates', async () => {
    const recent = parse(
      await tools['m9k_search']({ query: 'cors express', limit: 10, since: '3 days ago' }),
    );
    expect(recent.map((r) => r.chunkId).sort()).toEqual(['sess-copy:0', 'sess-live:0']);

    const similar = parse(
      await tools['m9k_similar_work']({
        description: 'cors express',
        limit: 5,
        timeframe: 'past 2 hours',
      }),
    );
    expect(similar.map((r) => r.chunkId)).toEqual(['sess-copy:0']);

    const sessions = parse(await tools['m9k_sessions']({ limit: 20, since: '2026-02-28' }));
    expect(sessions.map((s) => s.id).sort()).toEqual(['sess-copy', 'sess-live']);
  });

  it('a rejected date is a tool error, not a silent no-op', async () => {
    for (const [name, args] of [
      ['m9k_search', { query: 'x', limit: 5, timeframe: 'eventually' }],
      ['m9k_similar_work', { description: 'x', limit: 5, since: 'soon' }],
      ['m9k_sessions', { limit: 5, until: 'later' }],
    ] as const) {
      const r = (await tools[name](args)) as { isError?: boolean };
      expect(r.isError, name).toBe(true);
    }
  });

  it('manual saves are stamped in ISO-8601', async () => {
    await tools['m9k_save']({ content: 'deploy freeze starts friday' });
    const row = db
      .prepare(
        "SELECT timestamp FROM conv_chunks WHERE user_content = 'deploy freeze starts friday'",
      )
      .get() as { timestamp: string };
    expect(row.timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);
  });
});
