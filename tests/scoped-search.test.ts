import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type Database from 'better-sqlite3';
import { openMemoryDatabase, insertVectorsBatch } from '../src/db.js';
import { search, searchVectorized, searchFuzzy } from '../src/search.js';
import {
  parseAbsoluteDate,
  normalizeDateBound,
  resolveDateRange,
  DateInputError,
} from '../src/dates.js';
import { registerSearchTools } from '../src/tools/search.js';
import type { SearchContext, SearchResult } from '../src/models.js';
import { FakeEmbedder, captureTools } from './helpers/fixtures.js';

const embedder = new FakeEmbedder('fake-model', 384);

async function seedScoped(db: Database.Database): Promise<void> {
  db.prepare(
    `INSERT INTO conv_sessions (id, project, jsonl_path, started_at) VALUES
     ('big', '/big', '/t/big', '2026-01-01'), ('small', '/small', '/t/small', '2026-01-01')`,
  ).run();
  const ins = db.prepare(
    `INSERT INTO conv_chunks (id, session_id, idx, user_content, assistant_content, hash, timestamp)
     VALUES (?, ?, ?, ?, ?, ?, ?)`,
  );
  const items: Array<{ chunkId: string; embedding: Float32Array }> = [];
  for (let i = 0; i < 260; i++) {
    const text = 'alpha beta gamma delta';
    const day = String((i % 28) + 1).padStart(2, '0');
    ins.run(`big:${i}`, 'big', i, text, '', `hb${i}`, `2026-02-${day}T12:00:00.000Z`);
    items.push({ chunkId: `big:${i}`, embedding: await embedder.embed(text) });
  }
  ins.run('small:0', 'small', 0, 'alpha omega', '', 'hs0', '2026-03-10T00:00:00.000Z');
  items.push({ chunkId: 'small:0', embedding: await embedder.embed('alpha omega') });
  insertVectorsBatch(db, items);
}

const ctx: SearchContext = {
  embedderText: embedder,
  embedderCode: null,
  reranker: null,
  vecTextEnabled: true,
  vecCodeEnabled: false,
  autoFuzzyThreshold: 3,
};

const inScope = (r: SearchResult, f: { project?: string; since?: string; until?: string }) =>
  (!f.project || r.project === f.project) &&
  (!f.since || r.timestamp >= f.since) &&
  (!f.until || r.timestamp < f.until);

describe('filters before every cut (A4)', () => {
  let db: Database.Database;
  beforeEach(async () => {
    db = openMemoryDatabase().db;
    await seedScoped(db);
  });
  afterEach(() => db.close());

  it('finds a scoped vector match that sits past global rank 200', async () => {
    const q = await embedder.embedQuery('alpha beta');
    const global = searchVectorized(db, q, 250).map((r) => r.chunkId);
    expect(global.indexOf('small:0')).toBe(-1); // not even in the global top 250

    const scoped = searchVectorized(db, q, 20, 1.3, '_text', { project: '/small' });
    expect(scoped.map((r) => r.chunkId)).toEqual(['small:0']);
  });

  it('a nonexistent project returns nothing, without error', async () => {
    expect(await search(db, { query: 'alpha', limit: 10, project: '/none' }, ctx)).toEqual([]);
  });

  it('since is inclusive, until exclusive, and filters combine', async () => {
    const f = { since: '2026-02-10T12:00:00.000Z', until: '2026-02-12T12:00:00.000Z' };
    const r = await search(db, { query: 'alpha gamma', limit: 50, ...f }, ctx);
    expect(r.length).toBeGreaterThan(0);
    expect(r.every((x) => inScope(x, f))).toBe(true);
    expect(r.some((x) => x.timestamp === f.since)).toBe(true);
    expect(r.some((x) => x.timestamp === f.until)).toBe(false);

    const both = { project: '/small', since: '2026-03-01T00:00:00.000Z' };
    const r2 = await search(db, { query: 'alpha', limit: 10, ...both }, ctx);
    expect(r2.map((x) => x.chunkId)).toEqual(['small:0']);
  });

  it('every channel returns only in-scope rows, the fuzzy fallback included', async () => {
    const f = { project: '/small' };
    expect(searchFuzzy(db, 'alph', 10, f).map((r) => r.chunkId)).toEqual(['small:0']);
    const r = await search(db, { query: 'alph omeg', limit: 10, ...f }, ctx);
    expect(r.every((x) => inScope(x, f))).toBe(true);
  });
});

describe('date bounds (A4)', () => {
  it('reads plain dates and offset-less times as local time', () => {
    expect(normalizeDateBound('2026-10-13', 'since')).toBe(new Date(2026, 9, 13).toISOString());
    expect(normalizeDateBound('2026-10-13T08:30', 'since')).toBe(
      new Date(2026, 9, 13, 8, 30).toISOString(),
    );
  });

  it('honours explicit offsets', () => {
    expect(normalizeDateBound('2026-10-13T08:30:00Z', 'since')).toBe('2026-10-13T08:30:00.000Z');
    expect(normalizeDateBound('2026-10-13T08:30:00-07:00', 'since')).toBe(
      '2026-10-13T15:30:00.000Z',
    );
    expect(normalizeDateBound('2026-10-13 08:30:00.123456+0200', 'since')).toBe(
      '2026-10-13T06:30:00.123Z',
    );
  });

  it('rejects text it cannot parse, impossible dates and reversed ranges', () => {
    for (const bad of ['garbage', '2026-02-31', '2026-13-01', '10/13/2026', '']) {
      expect(parseAbsoluteDate(bad)).toBeNull();
    }
    expect(() => normalizeDateBound('2026-02-31', 'until')).toThrow(DateInputError);
    expect(() => resolveDateRange({ since: '2026-10-14', until: '2026-10-13' })).toThrow(
      DateInputError,
    );
  });

  it('m9k_search answers an unparseable date with a tool error', async () => {
    const db = openMemoryDatabase().db;
    const tools = captureTools(registerSearchTools, { db, searchContext: ctx });
    const res = (await tools['m9k_search']({ query: 'x', limit: 5, since: 'whenever' })) as {
      isError?: boolean;
      content: Array<{ text: string }>;
    };
    expect(res.isError).toBe(true);
    expect(JSON.parse(res.content[0].text).error).toMatch(/since/);
    db.close();
  });
});
