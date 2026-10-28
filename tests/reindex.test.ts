import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type Database from 'better-sqlite3';
import { openMemoryDatabase, getChunksWithoutEmbeddings, publishVectors } from '../src/db.js';
import { indexConvSession } from '../src/indexer.js';
import { FakeEmbedder, staleMappedVectors, vectorCounts } from './helpers/fixtures.js';

const SID = 'reindex-session';
type Exchange = [prompt: string, reply: string, files?: string[]];

/** A transcript with one user and one assistant line per exchange. */
function transcript(exchanges: Exchange[]): string {
  return exchanges
    .flatMap(([u, a, files = []], i) => {
      const t = (s: number) => new Date(Date.UTC(2026, 0, 1, 0, i, s)).toISOString();
      const tools = files.map((f, k) => ({
        type: 'tool_use',
        id: `t${i}-${k}`,
        name: 'Edit',
        input: { file_path: f },
      }));
      return [
        JSON.stringify({
          type: 'user',
          uuid: `u${i}`,
          sessionId: SID,
          cwd: '/proj',
          timestamp: t(0),
          message: { role: 'user', content: u },
        }),
        JSON.stringify({
          type: 'assistant',
          uuid: `a${i}`,
          parentUuid: `u${i}`,
          sessionId: SID,
          timestamp: t(1),
          message: { role: 'assistant', content: [{ type: 'text', text: a }, ...tools] },
        }),
      ];
    })
    .join('\n');
}

const ex = (n: number, files?: string[]): Exchange => [
  `question ${n}: why does the kafka consumer group keep rebalancing`,
  `answer ${n}: raise the session timeout and check the partition assignment`,
  files,
];

describe('incremental, transactional re-index (B1, B2)', () => {
  let db: Database.Database;
  const embedder = new FakeEmbedder();

  const index = (exchanges: Exchange[], mtimeMs?: number) => {
    const content = transcript(exchanges);
    return indexConvSession(
      db,
      SID,
      content,
      '/proj',
      '/t/s.jsonl',
      mtimeMs === undefined ? undefined : { mtimeMs },
    );
  };

  async function embedPending(): Promise<void> {
    const rows = getChunksWithoutEmbeddings(db, 1000, '_text');
    const vecs = await embedder.embedBatch(rows.map((r) => r.content));
    publishVectors(
      db,
      rows.map((r, i) => ({ chunkId: r.id, hash: r.hash, embedding: vecs[i] })),
      '_text',
    );
  }

  const vecRowids = () =>
    new Map(
      (
        db.prepare('SELECT chunk_id, vec_rowid FROM conv_vec_map_text').all() as Array<{
          chunk_id: string;
          vec_rowid: number;
        }>
      ).map((r) => [r.chunk_id, r.vec_rowid]),
    );
  const live = () =>
    db
      .prepare(
        'SELECT id, user_content FROM conv_chunks WHERE session_id = ? AND deleted_at IS NULL ORDER BY idx',
      )
      .all(SID) as Array<{ id: string; user_content: string }>;

  /** FTS matches its content table, foreign keys hold, one live chunk per vector, none stale. */
  async function checkInvariants(): Promise<void> {
    db.prepare(
      "INSERT INTO conv_chunks_fts(conv_chunks_fts, rank) VALUES ('integrity-check', 1)",
    ).run();
    expect(db.pragma('foreign_key_check')).toEqual([]);
    const c = vectorCounts(db, '_text');
    expect(c.vec).toBe(c.map);
    const bad = db
      .prepare(
        `SELECT COUNT(*) AS n FROM conv_vec_map_text m LEFT JOIN conv_chunks c ON c.id = m.chunk_id
         WHERE c.id IS NULL OR c.deleted_at IS NOT NULL`,
      )
      .get() as { n: number };
    expect(bad.n).toBe(0);
    expect(await staleMappedVectors(db, embedder)).toEqual([]);
  }

  beforeEach(() => {
    db = openMemoryDatabase().db;
  });
  afterEach(() => db.close());

  it('an appended exchange keeps every unchanged vector', async () => {
    index([ex(0), ex(1)], 1000);
    await embedPending();
    const before = vecRowids();

    expect(index([ex(0), ex(1), ex(2)], 2000)).toBe('indexed');
    const after = vecRowids();
    expect(after.get(`${SID}:0`)).toBe(before.get(`${SID}:0`));
    expect(after.get(`${SID}:1`)).toBe(before.get(`${SID}:1`));
    expect(after.has(`${SID}:2`)).toBe(false); // new, embedded on the next pass
    await checkInvariants();
  });

  it('a stale snapshot committed late is skipped', async () => {
    index([ex(0), ex(1), ex(2)], 2000);
    expect(index([ex(0), ex(1)], 1000)).toBe('skipped');
    expect(live()).toHaveLength(3);
    await checkInvariants();
  });

  it('a hash swap between positions does not abort, and drops both stale vectors', async () => {
    index([ex(0), ex(1)], 1000);
    await embedPending();
    expect(index([ex(1), ex(0)], 2000)).toBe('indexed');
    expect(live().map((r) => r.user_content)).toEqual([ex(1)[0], ex(0)[0]]);
    expect(vecRowids().size).toBe(0);
    await checkInvariants();
    await embedPending();
    await checkInvariants();
  });

  it('a prepended exchange cannot resurrect a forgotten one', async () => {
    index([ex(0), ex(1)], 1000);
    db.prepare("UPDATE conv_chunks SET deleted_at = '2026-01-02T00:00:00Z' WHERE id = ?").run(
      `${SID}:0`,
    );
    index([ex(9), ex(0), ex(1)], 2000);
    const tomb = db.prepare('SELECT deleted_at FROM conv_chunks WHERE id = ?').get(`${SID}:0`) as {
      deleted_at: string | null;
    };
    expect(tomb.deleted_at).not.toBeNull(); // the position stays tombstoned
    expect(live().map((r) => r.user_content)).toEqual([ex(1)[0]]); // forgotten text not back
    await checkInvariants();
  });

  it('a metadata-only change updates in place and keeps the vector', async () => {
    index([ex(0)], 1000);
    await embedPending();
    const before = vecRowids().get(`${SID}:0`);
    index([ex(0, ['/proj/src/a.ts'])], 2000);
    expect(vecRowids().get(`${SID}:0`)).toBe(before);
    const meta = db
      .prepare('SELECT metadata_json FROM conv_chunks WHERE id = ?')
      .get(`${SID}:0`) as {
      metadata_json: string;
    };
    expect(JSON.parse(meta.metadata_json).filePaths).toEqual(['/proj/src/a.ts']);
    await checkInvariants();
  });

  it('a crash mid-transaction leaves the previous index whole', async () => {
    index([ex(0), ex(1)], 1000);
    await embedPending();
    db.exec(`CREATE TEMP TRIGGER boom BEFORE INSERT ON conv_chunks WHEN NEW.idx = 3
             BEGIN SELECT RAISE(ABORT, 'injected crash'); END`);
    expect(() => index([ex(5), ex(1), ex(2), ex(3)], 2000)).toThrow(/injected crash/);
    expect(live().map((r) => r.user_content)).toEqual([ex(0)[0], ex(1)[0]]);
    expect(vecRowids().size).toBe(2);
    const s = db.prepare('SELECT file_mtime FROM conv_sessions WHERE id = ?').get(SID) as {
      file_mtime: number;
    };
    expect(s.file_mtime).toBe(1000);
    await checkInvariants();
  });

  it('50 growing re-index cycles leave no orphan or stale vector', async () => {
    const exchanges: Exchange[] = [];
    for (let i = 0; i < 50; i++) {
      exchanges.push(ex(i));
      // The last exchange grows each cycle (its reply is still being written).
      const growing: Exchange[] = [...exchanges];
      growing[i] = [ex(i)[0], `${ex(i)[1]} (draft ${i})`];
      index(growing, 1000 + i * 2);
      await embedPending();
      index(exchanges, 1001 + i * 2);
      await embedPending();
    }
    expect(live()).toHaveLength(50);
    const c = vectorCounts(db, '_text');
    expect(c).toEqual({ vec: 50, map: 50 });
    await checkInvariants();
  });

  it('an unchanged chunk keeps the time its row was written (pending age)', async () => {
    index([ex(0)], 1000);
    db.prepare("UPDATE conv_chunks SET indexed_at = '2026-01-01T00:00:00.000Z'").run();
    index([ex(0), ex(1)], 2000);
    const at = (id: string) =>
      (
        db.prepare('SELECT indexed_at FROM conv_chunks WHERE id = ?').get(id) as {
          indexed_at: string;
        }
      ).indexed_at;
    expect(at(`${SID}:0`)).toBe('2026-01-01T00:00:00.000Z');
    expect(at(`${SID}:1`) > '2026-01-02').toBe(true);
  });
});
