import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type Database from 'better-sqlite3';
import { openMemoryDatabase } from '../src/db.js';
import { chunkMessages, parseJSONL } from '../src/indexer.js';
import { ftsErrorCounts } from '../src/search.js';
import { registerSpecializedTools } from '../src/tools/specialized.js';
import { captureTools, seedSyntheticCorpus, type SyntheticChunk } from './helpers/fixtures.js';

const chunk = (
  id: string,
  timestamp: string,
  filePaths: string[],
  extra: Partial<SyntheticChunk> = {},
): SyntheticChunk => ({
  id,
  sessionId: 's',
  idx: 0,
  user: `exchange ${id}`,
  assistant: 'done',
  timestamp,
  metadata: { toolCalls: ['Edit'], filePaths },
  ...extra,
});

const CHUNKS: SyntheticChunk[] = [
  chunk('src-server', '2026-01-01T00:00:00Z', ['/proj/src/server.ts']),
  chunk('tests-server', '2026-01-05T00:00:00Z', ['/proj/tests/server.ts']),
  chunk('deleted', '2026-01-09T00:00:00Z', ['/proj/src/server.ts'], {
    deletedAt: '2026-01-10T00:00:00Z',
  }),
  chunk('underscore', '2026-01-02T00:00:00Z', ['/proj/tests/test_config.py']),
  chunk('underscore-trap', '2026-01-06T00:00:00Z', ['/proj/testXconfig.py']),
  chunk('percent', '2026-01-03T00:00:00Z', ['/proj/data/100%_done.csv']),
  chunk('percent-trap', '2026-01-07T00:00:00Z', ['/proj/data/100abc_done.csv']),
  chunk('escapes', '2026-01-04T00:00:00Z', ['/proj/odd "quoted" \\name.txt']),
  chunk('text-only', '2026-01-08T00:00:00Z', [], { user: 'settings live in test_config.py' }),
  ...[1, 2, 3, 4, 5].map((n) =>
    chunk(`index${n}`, `2026-02-0${n}T00:00:00Z`, [`/proj/pkg${n}/index.ts`]),
  ),
].map((c, i) => ({ ...c, idx: i }));

describe('m9k_file_history (A3)', () => {
  let db: Database.Database;
  let history: (filePath: string, limit?: number) => Promise<Array<Record<string, string>>>;

  beforeEach(() => {
    db = openMemoryDatabase().db;
    seedSyntheticCorpus(
      db,
      [{ id: 's', project: '/proj', jsonlPath: '/t/s', startedAt: '2026-01-01' }],
      CHUNKS,
    );
    const tools = captureTools(registerSpecializedTools, { db, searchContext: {} });
    history = async (filePath, limit = 10) =>
      JSON.parse((await tools['m9k_file_history']({ filePath, limit })).content[0].text) as Array<
        Record<string, string>
      >;
  });
  afterEach(() => db.close());

  const ids = (r: Array<Record<string, string>>) => r.map((x) => x.chunkId);

  it('ranks the matching path above a newer file with the same name elsewhere', async () => {
    const r = await history('src/server.ts');
    expect(r.slice(0, 2).map((x) => [x.chunkId, x.match])).toEqual([
      ['src-server', 'suffix'],
      ['tests-server', 'basename'],
    ]);
    expect(ids(r)).not.toContain('deleted');
  });

  it('an exact recorded path comes first', async () => {
    const r = await history('/proj/tests/server.ts');
    expect([r[0].chunkId, r[0].match]).toEqual(['tests-server', 'path']);
  });

  it('a basename-only call matches that name in any directory, newest first', async () => {
    expect(ids(await history('server.ts'))).toEqual(['tests-server', 'src-server']);
  });

  it('underscores are literal, and text mentions only fill the remaining slots', async () => {
    const r = await history('test_config.py');
    expect(r.map((x) => [x.chunkId, x.match])).toEqual([
      ['underscore', 'suffix'],
      ['text-only', 'text'],
    ]);
  });

  it('percent signs are literal', async () => {
    expect(ids(await history('data/100%_done.csv'))).toEqual(['percent']);
  });

  it('matches paths whose JSON form is escaped', async () => {
    const r = await history('/proj/odd "quoted" \\name.txt');
    expect([r[0].chunkId, r[0].match]).toEqual(['escapes', 'path']);
  });

  it('a directory matches the recorded paths under it', async () => {
    const r = await history('proj/src');
    expect(r.map((x) => [x.chunkId, x.match])).toEqual([['src-server', 'metadata']]);
  });

  it('a generic filename puts the given directory first and respects the limit', async () => {
    const r = await history('pkg3/index.ts', 3);
    expect(r).toHaveLength(3);
    expect([r[0].chunkId, r[0].match]).toEqual(['index3', 'suffix']);
    expect(r.slice(1).every((x) => x.match === 'basename')).toBe(true);
  });

  it('never throws or counts a syntax error on hostile names', async () => {
    const before = ftsErrorCounts.syntax;
    for (const name of ['"', '*', 'NEAR(a b)', 'a OR b', '../..', 'x"y.ts', '🎉.md', '   ']) {
      await expect(history(name)).resolves.toBeInstanceOf(Array);
    }
    expect(ftsErrorCounts.syntax).toBe(before);
  });
});

describe('patterns are kept apart from touched files (A3)', () => {
  it('Grep and Glob patterns go to metadata.patterns, not filePaths', () => {
    const line = (o: object) => JSON.stringify(o);
    const jsonl = [
      line({
        type: 'user',
        uuid: 'u1',
        timestamp: '2026-01-01T00:00:00Z',
        message: { role: 'user', content: 'find the TODOs' },
      }),
      line({
        type: 'assistant',
        uuid: 'a1',
        parentUuid: 'u1',
        timestamp: '2026-01-01T00:00:01Z',
        message: {
          role: 'assistant',
          content: [
            { type: 'tool_use', id: 't1', name: 'Grep', input: { pattern: 'TODO', path: 'src' } },
            { type: 'tool_use', id: 't2', name: 'Glob', input: { pattern: '**/*.ts' } },
            { type: 'tool_use', id: 't3', name: 'Read', input: { file_path: '/proj/a.ts' } },
            { type: 'text', text: 'Found two.' },
          ],
        },
      }),
    ].join('\n');
    const [c] = chunkMessages(parseJSONL(jsonl));
    expect(c.metadata.filePaths).toEqual(['/proj/a.ts']);
    expect(c.metadata.patterns).toEqual(['TODO', '**/*.ts']);
  });
});
