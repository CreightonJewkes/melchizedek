import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type Database from 'better-sqlite3';
import { openMemoryDatabase } from '../src/db.js';
import type { SearchContext } from '../src/models.js';
import { registerSearchTools } from '../src/tools/search.js';
import { registerSpecializedTools } from '../src/tools/specialized.js';
import { captureTools, FakeReranker, seedSyntheticCorpus } from './helpers/fixtures.js';

describe('m9k_errors skips the reranker (D34)', () => {
  let db: Database.Database;
  let reranker: FakeReranker;
  let tools: ReturnType<typeof captureTools>;

  beforeEach(() => {
    db = openMemoryDatabase().db;
    seedSyntheticCorpus(db);
    reranker = new FakeReranker();
    const searchContext: SearchContext = {
      embedderText: null,
      embedderCode: null,
      reranker,
      vecTextEnabled: false,
      vecCodeEnabled: false,
      autoFuzzyThreshold: 3,
    };
    const ctx = { db, cfg: {}, searchContext };
    tools = {
      ...captureTools(registerSearchTools, ctx),
      ...captureTools(registerSpecializedTools, ctx),
    };
  });
  afterEach(() => db.close());

  it('m9k_errors ranks by fusion alone; m9k_search still reranks', async () => {
    const errors = JSON.parse(
      (await tools['m9k_errors']({ errorMessage: 'error in the server', limit: 5 })).content[0]
        .text,
    ) as unknown[];
    expect(errors.length).toBeGreaterThan(0);
    expect(reranker.poolSizes).toEqual([]);

    await tools['m9k_search']({ query: 'error in the server', limit: 5 });
    expect(reranker.poolSizes.length).toBe(1);
  });
});
