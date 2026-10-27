import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type Database from 'better-sqlite3';
import { openMemoryDatabase } from '../src/db.js';
import {
  buildFtsQuery,
  searchBM25,
  search,
  ftsErrorCounts,
  MAX_QUERY_TERMS,
} from '../src/search.js';
import { seedSyntheticCorpus } from './helpers/fixtures.js';

const ids = (rs: Array<{ chunkId: string }>): string[] => rs.map((r) => r.chunkId).sort();

describe('keyword query builder (A1)', () => {
  let db: Database.Database;
  beforeEach(() => {
    db = openMemoryDatabase().db;
    seedSyntheticCorpus(db);
  });
  afterEach(() => db.close());

  const hostile = [
    "ModuleNotFoundError: No module named 'httpx'",
    'test_config.py',
    'multi-launch v2.1.0 -- foo',
    '"unclosed quote',
    'say ""hi"" there',
    'NOT AND OR NEAR',
    'col:value (paren) [bracket] {brace} ^caret *star +plus',
    'café 日本語 — naïve',
    '🎉🎉',
    '',
    '   ',
    '-',
    "it's a can't-miss",
    'x'.repeat(20_000),
    Array.from({ length: 1000 }, (_, i) => `t${i}`).join(' '),
  ];

  for (const q of hostile) {
    it(`never throws and is valid FTS5: ${JSON.stringify(q.slice(0, 40))}`, () => {
      const expr = buildFtsQuery(q);
      if (!expr) return; // nothing searchable: searchBM25 returns [] without querying
      expect(() =>
        db.prepare('SELECT COUNT(*) FROM conv_chunks_fts WHERE conv_chunks_fts MATCH ?').get(expr),
      ).not.toThrow();
      const before = ftsErrorCounts.syntax;
      searchBM25(db, expr, 10);
      expect(ftsErrorCounts.syntax).toBe(before);
    });
  }

  it('quotes each term as a phrase and joins with OR', () => {
    expect(buildFtsQuery('fix  CORS error')).toBe('"fix" OR "CORS" OR "error"');
    expect(buildFtsQuery('say "hi"')).toBe('"say" OR """hi"""');
    expect(buildFtsQuery('🎉 -- ok')).toBe('"ok"');
    expect(buildFtsQuery('🎉')).toBe('');
  });

  it('caps the number of terms as a safety bound', () => {
    const many = Array.from({ length: MAX_QUERY_TERMS + 50 }, (_, i) => `w${i}`).join(' ');
    expect(buildFtsQuery(many).split(' OR ')).toHaveLength(MAX_QUERY_TERMS);
  });

  it('finds an error string that used to be a syntax error', () => {
    const r = searchBM25(db, buildFtsQuery("ModuleNotFoundError: No module named 'httpx'"), 10);
    expect(ids(r)).toContain('sess-live:1');
  });

  it('keeps dotted and underscored names as one phrase', () => {
    const r = searchBM25(db, buildFtsQuery('test_config.py'), 10);
    expect(r).toEqual([]); // only metadata mentions it; the phrase needs adjacent tokens
    const hit = searchBM25(db, buildFtsQuery('Express server?'), 10);
    expect(ids(hit)).toEqual(['sess-copy:0', 'sess-live:0']);
  });

  it('matches accented text and operator words literally', () => {
    expect(ids(searchBM25(db, buildFtsQuery('café'), 10))).toEqual(['sess-live:2']);
    expect(searchBM25(db, buildFtsQuery('NOT NEAR'), 10)).toEqual([]);
  });

  it('OR means one matching word is enough (no longer implicit AND)', () => {
    const r = searchBM25(db, buildFtsQuery('httpx zzznomatch'), 10);
    expect(ids(r)).toEqual(['sess-live:1']);
  });

  it('counts syntax failures separately and still returns empty', () => {
    const before = { ...ftsErrorCounts };
    expect(searchBM25(db, '"unbalanced', 10)).toEqual([]);
    expect(ftsErrorCounts.syntax).toBe(before.syntax + 1);
    expect(ftsErrorCounts.busy).toBe(before.busy);
  });

  it('search() returns results for the hostile inputs that have words', async () => {
    const r = await search(db, { query: "No module named 'httpx'", limit: 5 });
    expect(r.map((x) => x.chunkId)).toContain('sess-live:1');
    expect(await search(db, { query: '🎉', limit: 5 })).toEqual([]);
  });
});
