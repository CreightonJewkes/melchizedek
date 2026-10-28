import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type Database from 'better-sqlite3';
import {
  openMemoryDatabase,
  getChunksWithoutEmbeddings,
  publishVectors,
  deleteVectorsForChunks,
  sweepOrphanVectors,
  deadVectorShare,
  maintainVectors,
  setMeta,
  getMeta,
} from '../src/db.js';
import { searchVectorized } from '../src/search.js';
import { FakeEmbedder, seedSyntheticCorpus, vectorCounts } from './helpers/fixtures.js';

describe('orphan sweep and compaction (B2b)', () => {
  let db: Database.Database;
  const embedder = new FakeEmbedder();
  const blob = () => Buffer.from(new Float32Array(384).fill(0.05).buffer);

  beforeEach(async () => {
    db = openMemoryDatabase().db;
    seedSyntheticCorpus(db);
    const rows = getChunksWithoutEmbeddings(db, 100, '_text');
    const vecs = await embedder.embedBatch(rows.map((r) => r.content));
    publishVectors(
      db,
      rows.map((r, i) => ({ chunkId: r.id, hash: r.hash, embedding: vecs[i] })),
      '_text',
    );
  });
  afterEach(() => db.close());

  const mapped = () =>
    db.prepare('SELECT chunk_id, vec_rowid FROM conv_vec_map_text ORDER BY chunk_id').all();

  it('removes orphan vectors and dangling maps, and keeps every live vector', async () => {
    const before = mapped();
    const live = before.length;
    // 30 vec0 rows nothing points to (what 1.0.2's re-index left behind)
    const ins = db.prepare('INSERT INTO conv_vec_text (embedding) VALUES (?)');
    for (let i = 0; i < 30; i++) ins.run(blob());
    // a map row whose vector is gone, and a vector mapped to a forgotten chunk
    const [first, second] = before as Array<{ chunk_id: string; vec_rowid: number }>;
    db.prepare('DELETE FROM conv_vec_text WHERE rowid = ?').run(first.vec_rowid);
    db.prepare("UPDATE conv_chunks SET deleted_at = '2026-03-01T00:00:00Z' WHERE id = ?").run(
      second.chunk_id,
    );

    const r = sweepOrphanVectors(db, 7); // a small batch, so the loop runs several times
    expect(r).toEqual({ orphanVectors: 30, danglingMaps: 2, skipped: [] });
    expect(vectorCounts(db, '_text')).toEqual({ vec: live - 2, map: live - 2 });
    expect(mapped()).toEqual(
      before.filter((m) => m !== first && m !== second).map((m) => ({ ...(m as object) })),
    );
    expect(sweepOrphanVectors(db)).toEqual({ orphanVectors: 0, danglingMaps: 0, skipped: [] });
  });

  it('leaves a generation alone while its migration is in progress', () => {
    db.prepare('INSERT INTO conv_vec_text (embedding) VALUES (?)').run(blob());
    setMeta(db, 'migration_target_model_text', 'other-model');
    expect(sweepOrphanVectors(db).skipped).toEqual(['conv_vec_text']);
    const c = vectorCounts(db, '_text');
    expect(c.vec).toBe(c.map + 1);
  });

  it('counts deletions, and compacts a table mostly made of dead slots without changing results', async () => {
    const ins = db.prepare('INSERT INTO conv_vec_text (embedding) VALUES (?)');
    for (let i = 0; i < 40; i++) ins.run(blob());
    const q = await embedder.embedQuery('cors middleware express');
    const ranked = searchVectorized(db, q, 5).map((r) => [r.chunkId, r.score.toFixed(6)]);
    const before = mapped();

    sweepOrphanVectors(db);
    expect(getMeta(db, 'vec_dead_rows:conv_vec_text')).toBe('40');
    expect(deadVectorShare(db, 'conv_vec_text')).toBeGreaterThan(0.5);

    const m = maintainVectors(db);
    expect(m.compacted).toEqual(['conv_vec_text']);
    expect(getMeta(db, 'vec_dead_rows:conv_vec_text')).toBeNull();
    expect(mapped()).toEqual(before); // rowids kept, so the map is untouched
    expect(searchVectorized(db, q, 5).map((r) => [r.chunkId, r.score.toFixed(6)])).toEqual(ranked);
    expect(maintainVectors(db).compacted).toEqual([]); // nothing dead any more
  });

  it('the shared deletion path counts what it removes', () => {
    const ids = (mapped() as Array<{ chunk_id: string }>).slice(0, 2).map((m) => m.chunk_id);
    deleteVectorsForChunks(db, ids);
    expect(getMeta(db, 'vec_dead_rows:conv_vec_text')).toBe('2');
  });
});
