import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type Database from 'better-sqlite3';
import { openMemoryDatabase, getMeta, getVecTableDimensions } from '../src/db.js';
import { embedChunks, indexConvSession } from '../src/indexer.js';
import {
  EmbeddingMigration,
  seedEmbeddingMeta,
  detectModelChange,
  vectorIndexMatches,
  reconcileVecTables,
} from '../src/migration.js';
import { search } from '../src/search.js';
import type { SearchContext } from '../src/models.js';
import {
  FakeEmbedder,
  seedSyntheticCorpus,
  vectorCounts,
  staleMappedVectors,
} from './helpers/fixtures.js';

const tables = (db: Database.Database): string[] =>
  db.prepare("SELECT name FROM sqlite_master WHERE type = 'table'").pluck().all() as string[];

const eligible = (db: Database.Database): number =>
  (
    db
      .prepare(
        `SELECT COUNT(*) AS n FROM conv_chunks WHERE deleted_at IS NULL
         AND LENGTH(user_content) + LENGTH(assistant_content) >= 50`,
      )
      .get() as { n: number }
  ).n;

/** A database whose text index was built by `old`. */
async function indexedWith(old: FakeEmbedder): Promise<Database.Database> {
  const db = openMemoryDatabase().db;
  seedSyntheticCorpus(db);
  seedEmbeddingMeta(db, old, '_text');
  await embedChunks(db, old, '_text', 50);
  return db;
}

function ctxFor(embedder: FakeEmbedder): SearchContext {
  return {
    embedderText: embedder,
    embedderCode: null,
    reranker: null,
    vecTextEnabled: true,
    vecCodeEnabled: false,
    autoFuzzyThreshold: 0,
  };
}

/** Make the Nth statement matching `pattern` (exec or meta write) throw. */
function injectFault(
  db: Database.Database,
  pattern: string | ((sql: string, args: unknown[]) => boolean),
): () => void {
  const exec = db.exec.bind(db);
  const prepare = db.prepare.bind(db);
  const hit = (sql: string, args: unknown[] = []): boolean =>
    typeof pattern === 'string' ? sql.includes(pattern) : pattern(sql, args);
  db.exec = ((sql: string) => {
    if (hit(sql)) throw new Error(`injected fault at: ${sql.trim().slice(0, 60)}`);
    return exec(sql);
  }) as typeof db.exec;
  db.prepare = ((sql: string) => {
    const stmt = prepare(sql);
    if (!sql.startsWith('INSERT OR REPLACE INTO meta')) return stmt;
    return new Proxy(stmt, {
      get(target, prop, recv) {
        if (prop !== 'run') return Reflect.get(target, prop, recv);
        return (...args: unknown[]) => {
          if (hit(sql, args)) throw new Error(`injected fault at meta ${String(args[0])}`);
          return target.run(...(args as []));
        };
      },
    });
  }) as typeof db.prepare;
  return () => {
    db.exec = exec;
    db.prepare = prepare;
  };
}

const SWAP_STEPS: Array<[string, string | ((sql: string, args: unknown[]) => boolean)]> = [
  ['drop old tables', 'DROP TABLE IF EXISTS conv_vec_map_text;'],
  ['create new vec table', 'CREATE VIRTUAL TABLE IF NOT EXISTS conv_vec_text USING'],
  ['copy vectors', 'INSERT INTO conv_vec_text(rowid, embedding)'],
  ['drop next tables', 'DROP TABLE IF EXISTS conv_vec_map_text_next;'],
  [
    'write model meta',
    (sql, args) => sql.includes('meta') && args[0] === 'embedding_model_id_text',
  ],
  ['write recipe meta', (sql, args) => sql.includes('meta') && args[0] === 'embedding_recipe_text'],
];

describe('safe model migration (L2)', () => {
  const oldModel = new FakeEmbedder('old-model', 384);
  const newModel = new FakeEmbedder('new-model', 384);
  let db: Database.Database;

  beforeEach(async () => {
    db = await indexedWith(oldModel);
  });
  afterEach(() => db.close());

  for (const [step, pattern] of SWAP_STEPS) {
    it(`a fault at "${step}" leaves the old generation searchable and the migration resumable`, async () => {
      const before = vectorCounts(db, '_text');
      const migration = new EmbeddingMigration(db, newModel);
      migration.prepare();
      await migration.reembed();

      const restore = injectFault(db, pattern);
      expect(() => migration.swap()).toThrow('injected fault');
      restore();

      expect(vectorCounts(db, '_text')).toEqual(before);
      expect(await staleMappedVectors(db, oldModel)).toEqual([]);
      expect(getMeta(db, 'embedding_model_id_text')).toBe('old-model');
      expect(tables(db)).toContain('conv_vec_map_text_next');
      expect(getMeta(db, 'migration_target_model_text')).toBe('new-model');
      // Mid-migration, search must not compare new-model queries with old-model vectors.
      expect(vectorIndexMatches(db, newModel)).toBe(false);

      // The next run resumes and publishes.
      const result = await EmbeddingMigration.resume(db, newModel)!.run();
      expect(result.swapped).toBe(true);
      expect(getMeta(db, 'embedding_model_id_text')).toBe('new-model');
      expect(await staleMappedVectors(db, newModel)).toEqual([]);
      expect(vectorCounts(db, '_text').map).toBe(eligible(db));
    });
  }

  it('a failed swap inside run() reports failure and keeps the old index', async () => {
    const restore = injectFault(db, 'INSERT INTO conv_vec_text(rowid, embedding)');
    const result = await new EmbeddingMigration(db, newModel).run();
    restore();
    expect(result).toMatchObject({ swapped: false, aborted: true });
    expect(getMeta(db, 'embedding_model_id_text')).toBe('old-model');
    expect(await staleMappedVectors(db, oldModel)).toEqual([]);
    expect(vectorIndexMatches(db, oldModel)).toBe(true);
    expect(vectorIndexMatches(db, newModel)).toBe(false);
    expect(detectModelChange(db, newModel)).not.toBeNull(); // retried next start
  });

  it('same-dimension switch: keyword-only while migrating, vectors from the new model after', async () => {
    const vectorHits = async (): Promise<number> =>
      (await search(db, { query: 'cors express middleware', limit: 10 }, ctxFor(newModel))).filter(
        (r) => r.matchType !== 'bm25' && r.matchType !== 'fuzzy',
      ).length;

    expect(vectorIndexMatches(db, newModel)).toBe(false); // stored index is old-model
    expect(await vectorHits()).toBe(0);

    const migration = new EmbeddingMigration(db, newModel, { batchSize: 2 });
    migration.prepare();
    await migration.reembedBatch();
    expect(await vectorHits()).toBe(0); // still migrating

    const result = await EmbeddingMigration.resume(db, newModel)!.run();
    expect(result.swapped).toBe(true);
    expect(vectorIndexMatches(db, newModel)).toBe(true);
    expect(await vectorHits()).toBeGreaterThan(0);
    expect(await staleMappedVectors(db, newModel)).toEqual([]);
  });

  it('dimension-changing switch keeps the old tables at startup and replaces them atomically', async () => {
    const wide = new FakeEmbedder('wide-model', 768);
    const before = vectorCounts(db, '_text');
    reconcileVecTables(db, wide, '_text'); // what the daemon does at startup
    expect(vectorCounts(db, '_text')).toEqual(before);
    expect(getVecTableDimensions(db, '_text')).toBe(384);
    expect(vectorIndexMatches(db, wide)).toBe(false);

    const result = await new EmbeddingMigration(db, wide).run();
    expect(result.swapped).toBe(true);
    expect(getVecTableDimensions(db, '_text')).toBe(768);
    expect(await staleMappedVectors(db, wide)).toEqual([]);
    const hits = await search(db, { query: 'cors express', limit: 5 }, ctxFor(wide));
    expect(hits.some((r) => r.matchType === 'hybrid' || r.matchType === 'vector_text')).toBe(true);
  });

  it('a hook rewriting a session during the migration leaves no stale or orphan vectors', async () => {
    const lines = (turns: string[]): string =>
      turns
        .flatMap((t, i) => [
          JSON.stringify({
            type: 'user',
            uuid: `u${i}`,
            sessionId: 'sess-live',
            timestamp: `2026-03-01T11:0${i}:00.000Z`,
            cwd: '/proj/alpha',
            message: { role: 'user', content: t },
          }),
          JSON.stringify({
            type: 'assistant',
            uuid: `a${i}`,
            sessionId: 'sess-live',
            timestamp: `2026-03-01T11:0${i}:30.000Z`,
            cwd: '/proj/alpha',
            message: { role: 'assistant', content: [{ type: 'text', text: `reply about ${t}` }] },
          }),
        ])
        .join('\n');
    let fired = false;
    newModel.onBatch = () => {
      if (fired) return;
      fired = true;
      indexConvSession(
        db,
        'sess-live',
        lines(['rewritten first exchange about ingress', 'brand new second exchange text']),
        '/proj/alpha',
        '/tmp/sess-live.jsonl',
      );
    };
    const result = await new EmbeddingMigration(db, newModel, { batchSize: 2 }).run();
    newModel.onBatch = undefined;
    expect(result.swapped).toBe(true);
    expect(await staleMappedVectors(db, newModel)).toEqual([]);
    const c = vectorCounts(db, '_text');
    expect(c.vec).toBe(c.map);
  });

  it('resumes after an interrupted run without re-embedding or duplicating', async () => {
    const controller = new AbortController();
    let batches = 0;
    newModel.onBatch = () => {
      if (++batches === 1) controller.abort();
    };
    const first = await new EmbeddingMigration(db, newModel, {
      batchSize: 2,
      signal: controller.signal,
    }).run();
    newModel.onBatch = undefined;
    expect(first).toMatchObject({ swapped: false, aborted: true });
    expect(tables(db)).toContain('conv_vec_map_text_next'); // progress kept
    expect(getMeta(db, 'migration_lock_text')).toBeNull();
    const done = vectorCounts(db, '_text_next').map;
    expect(done).toBe(2);

    newModel.calls = [];
    const second = await EmbeddingMigration.resume(db, newModel, { batchSize: 2 })!.run();
    expect(second.swapped).toBe(true);
    const reembedded = newModel.calls.flat().length;
    expect(reembedded).toBe(eligible(db) - done);
    expect(vectorCounts(db, '_text').map).toBe(eligible(db));
    const c = vectorCounts(db, '_text');
    expect(c.vec).toBe(c.map);
  });

  it('a recipe change counts as a model change', () => {
    const sameIdNewRecipe = new FakeEmbedder('old-model', 384);
    sameIdNewRecipe.recipe = () => '{"query":"query: "}';
    seedEmbeddingMeta(db, oldModel, '_text');
    expect(detectModelChange(db, oldModel)).toBeNull();
    expect(detectModelChange(db, sameIdNewRecipe)).not.toBeNull();
    expect(vectorIndexMatches(db, sameIdNewRecipe)).toBe(false);
  });
});
