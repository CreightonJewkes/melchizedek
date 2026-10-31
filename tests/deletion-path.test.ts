import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { fork } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';
import Database from 'better-sqlite3';
import {
  openDatabase,
  openMemoryDatabase,
  deleteChunksWhere,
  deleteSessionsWhere,
  drainPendingVecCleanup,
  ignoreProject,
} from '../src/db.js';
import {
  indexConvSession,
  detectOrphanedSessions,
  backfillExistingSessions,
} from '../src/indexer.js';
import { registerManageTools } from '../src/tools/manage.js';
import {
  seedSyntheticCorpus,
  seedVectors,
  vectorCounts,
  captureTools,
  tempDbPath,
} from './helpers/fixtures.js';

const GENERATIONS = ['_text', '_code', '_text_next'];
const LIVE_CHUNKS = ['sess-live:0', 'sess-live:1', 'sess-live:2', 'sess-archive:0', 'sess-copy:0'];

function jsonl(sessionId: string, cwd: string, turns: string[]): string {
  return turns
    .flatMap((t, i) => [
      JSON.stringify({
        type: 'user',
        uuid: `u${i}`,
        sessionId,
        timestamp: `2026-04-01T10:0${i}:00.000Z`,
        cwd,
        message: { role: 'user', content: t },
      }),
      JSON.stringify({
        type: 'assistant',
        uuid: `a${i}`,
        sessionId,
        timestamp: `2026-04-01T10:0${i}:30.000Z`,
        cwd,
        message: { role: 'assistant', content: [{ type: 'text', text: `answer to ${t}` }] },
      }),
    ])
    .join('\n');
}

describe('one deletion path (L4)', () => {
  let db: Database.Database;
  let tools: ReturnType<typeof captureTools>;

  beforeEach(() => {
    db = openMemoryDatabase().db;
    seedSyntheticCorpus(db);
    seedVectors(db, GENERATIONS, LIVE_CHUNKS);
    // Code search is disabled for queries: its vectors must still be cleaned up.
    tools = captureTools(registerManageTools, {
      db,
      searchContext: { vecTextEnabled: true, vecCodeEnabled: false },
    });
  });

  afterEach(() => db.close());

  const expectNoOrphans = (): void => {
    for (const g of GENERATIONS) {
      const c = vectorCounts(db, g);
      expect(c.vec, `orphans in ${g}`).toBe(c.map);
    }
  };

  it('forget removes the chunk vectors from every generation and keeps a tombstone', async () => {
    await tools['m9k_forget']({ chunkId: 'sess-live:0' });
    expectNoOrphans();
    for (const g of GENERATIONS) expect(vectorCounts(db, g).map).toBe(4);
    const row = db.prepare("SELECT deleted_at FROM conv_chunks WHERE id = 'sess-live:0'").get() as {
      deleted_at: string | null;
    };
    expect(row.deleted_at).not.toBeNull();
  });

  it('delete-session removes chunks and vectors and tombstones the session', async () => {
    await tools['m9k_delete_session']({ sessionId: 'sess-live' });
    expectNoOrphans();
    for (const g of GENERATIONS) expect(vectorCounts(db, g).map).toBe(2);
  });

  it('project purge removes every session, chunk and vector of the project', async () => {
    await tools['m9k_ignore_project']({ project: '/proj/alpha', purge: true });
    expectNoOrphans();
    for (const g of GENERATIONS) expect(vectorCounts(db, g).map).toBe(1); // sess-copy only
    const left = db
      .prepare("SELECT COUNT(*) AS n FROM conv_sessions WHERE project = '/proj/alpha'")
      .get();
    expect(left).toEqual({ n: 0 });
  });

  it('re-indexing a changed transcript leaves no orphan vectors', () => {
    indexConvSession(db, 's-re', jsonl('s-re', '/p', ['one', 'two']), '/p', '/t/s-re.jsonl');
    seedVectors(db, GENERATIONS, ['s-re:0', 's-re:1']);
    indexConvSession(
      db,
      's-re',
      jsonl('s-re', '/p', ['one', 'two', 'three']),
      '/p',
      '/t/s-re.jsonl',
    );
    expectNoOrphans();
  });

  it('orphan-session purge removes their vectors; without purge the archive is kept', () => {
    const kept = detectOrphanedSessions(db, false);
    expect(kept.purgedCount).toBe(0);
    expect(db.prepare("SELECT 1 AS x FROM conv_sessions WHERE id = 'sess-archive'").get()).toEqual({
      x: 1,
    });

    detectOrphanedSessions(db, true);
    expectNoOrphans();
  });

  it('ghost-session cleanup goes through the same path', () => {
    db.prepare(
      "INSERT INTO conv_sessions (id, project, jsonl_path, started_at) VALUES ('ghost', '/p', '/t/ghost.jsonl', '2026-01-01')",
    ).run();
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'm9k-ghost-'));
    backfillExistingSessions(db, dir);
    fs.rmSync(dir, { recursive: true, force: true });
    expect(db.prepare("SELECT COUNT(*) AS n FROM conv_sessions WHERE id = 'ghost'").get()).toEqual({
      n: 0,
    });
    expectNoOrphans();
  });

  it('indexConvSession refuses ignored projects, whichever hook calls it', () => {
    ignoreProject(db, '/secret');
    const status = indexConvSession(
      db,
      's-sec',
      jsonl('s-sec', '/secret', ['x']),
      '/secret',
      '/t/s.jsonl',
    );
    expect(status).toBe('skipped');
    expect(db.prepare("SELECT COUNT(*) AS n FROM conv_sessions WHERE id = 's-sec'").get()).toEqual({
      n: 0,
    });
  });
});

describe('deletion without sqlite-vec (L4)', () => {
  let cleanup = (): void => {};
  afterEach(() => cleanup());

  it('queues vec0 rowids instead of failing, and the daemon drains them', () => {
    const tmp = tempDbPath();
    cleanup = tmp.cleanup;
    const daemonSide = openDatabase(tmp.dbPath).db;
    seedSyntheticCorpus(daemonSide);
    seedVectors(daemonSide, ['_text'], LIVE_CHUNKS);

    // A hook process that could not load sqlite-vec.
    const hook = new Database(tmp.dbPath);
    hook.pragma('foreign_keys = ON');
    const removed = deleteChunksWhere(hook, "session_id = 'sess-live'");
    expect(removed).toBe(3);
    const queued = hook.prepare('SELECT COUNT(*) AS n FROM pending_vec_cleanup').get();
    expect(queued).toEqual({ n: 3 });
    hook.close();

    expect(vectorCounts(daemonSide, '_text')).toEqual({ vec: 5, map: 2 });
    expect(drainPendingVecCleanup(daemonSide)).toBe(3);
    expect(vectorCounts(daemonSide, '_text')).toEqual({ vec: 2, map: 2 });
    expect(daemonSide.prepare('SELECT COUNT(*) AS n FROM pending_vec_cleanup').get()).toEqual({
      n: 0,
    });
    daemonSide.close();
  });

  it('waits out another process holding the write lock instead of failing at once', async () => {
    const tmp = tempDbPath();
    cleanup = tmp.cleanup;
    const daemonSide = openDatabase(tmp.dbPath).db;
    seedSyntheticCorpus(daemonSide);
    seedVectors(daemonSide, ['_text'], LIVE_CHUNKS);

    // A hook process re-indexing holds the write lock for a moment.
    const holder = path.join(path.dirname(tmp.dbPath), 'hold.mjs');
    const pkg = path.join(import.meta.dirname, '..', 'package.json');
    fs.writeFileSync(
      holder,
      `import { createRequire } from 'module';
       const Database = createRequire(${JSON.stringify(pkg)})('better-sqlite3');
       const db = new Database(process.argv[2]);
       db.exec('BEGIN IMMEDIATE');
       process.send('locked');
       setTimeout(() => { db.exec('COMMIT'); db.close(); process.exit(0); }, 300);`,
    );
    const child = fork(holder, [tmp.dbPath]);
    const exited = new Promise((resolve) => child.once('exit', resolve));
    await new Promise((resolve) => child.once('message', resolve));

    // A deferred transaction reads first, then fails to upgrade with SQLITE_BUSY at once.
    expect(deleteSessionsWhere(daemonSide, "id = 'sess-live'")).toEqual({
      sessions: 1,
      chunks: 3,
    });
    expect(vectorCounts(daemonSide, '_text')).toEqual({ vec: 2, map: 2 });
    await exited;
    daemonSide.close();
  }, 15_000);
});
