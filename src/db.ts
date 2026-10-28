/**
 * SQLite layer: schema init, CRUD, WAL mode.
 */

import fs from 'fs';
import path from 'path';
import Database from 'better-sqlite3';
import { logger } from './logger.js';

export type { Database as DatabaseType } from 'better-sqlite3';

const P = 'db';

export interface DatabaseInfo {
  db: Database.Database;
  vecEnabled: boolean;
  /** False if schema migration was needed but couldn't acquire the lock. */
  schemaReady: boolean;
}

export function loadSqliteVec(db: Database.Database): boolean {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const sqliteVec = require('sqlite-vec');
    sqliteVec.load(db);
    return true;
  } catch {
    logger.debug(P, 'sqlite-vec not available — vector search disabled');
    return false;
  }
}

export function openDatabase(dbPath: string): DatabaseInfo {
  fs.mkdirSync(path.dirname(dbPath), { recursive: true });
  const db = new Database(dbPath);
  db.pragma('busy_timeout = 5000');
  // Refuse a database written by newer code before touching it in any way.
  try {
    assertSchemaNotNewer(db);
  } catch (err) {
    db.close();
    throw err;
  }
  db.pragma('journal_mode = WAL');
  db.pragma('foreign_keys = ON');
  const vecEnabled = loadSqliteVec(db);
  const schemaReady = initSchema(db, vecEnabled);
  return { db, vecEnabled, schemaReady };
}

export function openMemoryDatabase(): DatabaseInfo {
  const db = new Database(':memory:');
  db.pragma('foreign_keys = ON');
  const vecEnabled = loadSqliteVec(db);
  const schemaReady = initSchema(db, vecEnabled);
  return { db, vecEnabled, schemaReady };
}

export const SCHEMA_VERSION = 3;

/**
 * The database is NOT a cache: most archived sessions no longer have a transcript on
 * disk, so it is the only copy of that history. Schema changes are therefore additive
 * steps keyed by user_version. A step may create tables, indexes and columns, or fix
 * data in place; it never drops or recreates sessions, chunks, memories or tombstones.
 *
 * Step N upgrades user_version N-1 to N. `ddl` (virtual tables, which SQLite cannot
 * create inside every transaction) runs first and must be idempotent; `run` and the
 * version bump then commit together in one IMMEDIATE transaction, so a failure leaves
 * the database openable at the previous version.
 */
export interface SchemaStep {
  version: number;
  name: string;
  ddl?: (db: Database.Database) => void;
  run: (db: Database.Database) => void;
}

export const SCHEMA_STEPS: readonly SchemaStep[] = [
  {
    version: 3,
    name: 'pending vector cleanup queue; ISO-8601 chunk timestamps',
    run: (db) => {
      db.exec(`
        CREATE TABLE IF NOT EXISTS pending_vec_cleanup (
          vec_table TEXT NOT NULL,
          vec_rowid INTEGER NOT NULL,
          queued_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
          PRIMARY KEY (vec_table, vec_rowid)
        );
      `);
      // Manual saves wrote datetime('now') (a space, UTC). Lexical date filters compare
      // against ISO bounds, so rewrite those few rows in the T form everything else uses.
      db.prepare(
        `UPDATE conv_chunks SET timestamp = strftime('%Y-%m-%dT%H:%M:%fZ', timestamp)
         WHERE timestamp GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:*'`,
      ).run();
    },
  },
];

export class SchemaVersionError extends Error {}

function userVersion(db: Database.Database): number {
  return db.pragma('user_version', { simple: true }) as number;
}

function assertSchemaNotNewer(db: Database.Database): void {
  const version = userVersion(db);
  if (version > SCHEMA_VERSION) {
    throw new SchemaVersionError(
      `Database schema version ${version} is newer than this melchizedek supports (${SCHEMA_VERSION}). ` +
        'Upgrade melchizedek; the database was left untouched.',
    );
  }
}

function isBusy(err: unknown): boolean {
  const code = (err as { code?: string }).code;
  return code === 'SQLITE_BUSY' || code === 'SQLITE_LOCKED';
}

/**
 * Bring the schema to SCHEMA_VERSION without destroying data. Idempotent and safe
 * when several processes open the database at once (each step re-checks the version
 * inside its write transaction). Returns false when another writer holds the lock
 * past busy_timeout; callers then skip work and retry on the next open.
 */
export function initSchema(
  db: Database.Database,
  vecEnabled: boolean,
  steps: readonly SchemaStep[] = SCHEMA_STEPS,
): boolean {
  assertSchemaNotNewer(db);
  const target = steps.length > 0 ? steps[steps.length - 1].version : 2;
  if (userVersion(db) >= target) return true;

  try {
    if (userVersion(db) < 2) {
      if (userVersion(db) === 1) {
        // Pre-0.9.1 layout. Its tables are kept untouched beside the new ones.
        logger.warn(P, 'Schema version 1 found: legacy tables kept; creating current schema');
      }
      createBaseSchema(db);
      db.transaction(() => {
        if (userVersion(db) < 2) db.pragma('user_version = 2');
      }).immediate();
      if (vecEnabled) createVecTables(db, 384, '_text');
    }

    for (const step of steps) {
      if (userVersion(db) >= step.version) continue;
      step.ddl?.(db);
      db.transaction(() => {
        if (userVersion(db) >= step.version) return; // another process got there first
        step.run(db);
        db.pragma(`user_version = ${step.version}`);
      }).immediate();
      logger.info(P, `Schema upgraded to version ${step.version}: ${step.name}`);
    }
    return true;
  } catch (err) {
    if (isBusy(err)) {
      logger.warn(P, 'Schema upgrade deferred: database is locked by another writer');
      return false;
    }
    throw err;
  }
}

/** The version-2 tables (unchanged from 1.0.2). Idempotent. */
function createBaseSchema(db: Database.Database): void {
  db.exec(`
    -- Conversation sessions
    CREATE TABLE IF NOT EXISTS conv_sessions (
      id TEXT PRIMARY KEY,
      project TEXT NOT NULL,
      jsonl_path TEXT NOT NULL,
      file_hash TEXT NOT NULL DEFAULT '',
      file_size INTEGER NOT NULL DEFAULT 0,
      started_at TEXT NOT NULL,
      ended_at TEXT,
      message_count INTEGER NOT NULL DEFAULT 0,
      chunk_count INTEGER NOT NULL DEFAULT 0,
      indexed_at TEXT NOT NULL DEFAULT (datetime('now')),
      deleted_at TEXT,
      UNIQUE(jsonl_path)
    );

    CREATE INDEX IF NOT EXISTS idx_conv_sessions_project ON conv_sessions(project);
    CREATE INDEX IF NOT EXISTS idx_conv_sessions_started_at ON conv_sessions(started_at);

    -- Conversation chunks
    CREATE TABLE IF NOT EXISTS conv_chunks (
      id TEXT PRIMARY KEY,
      session_id TEXT NOT NULL REFERENCES conv_sessions(id) ON DELETE CASCADE,
      idx INTEGER NOT NULL,
      kind TEXT NOT NULL DEFAULT 'exchange', -- see CONV_KIND_EXCHANGE in constants.ts
      user_content TEXT NOT NULL,
      assistant_content TEXT NOT NULL DEFAULT '',
      hash TEXT NOT NULL,
      timestamp TEXT NOT NULL,
      token_count INTEGER,
      tags TEXT,
      metadata_json TEXT NOT NULL DEFAULT '{}',
      embedding BLOB,
      deleted_at TEXT,
      UNIQUE(session_id, hash)
    );

    CREATE INDEX IF NOT EXISTS idx_conv_chunks_session ON conv_chunks(session_id);
    CREATE INDEX IF NOT EXISTS idx_conv_chunks_hash ON conv_chunks(hash);
    CREATE INDEX IF NOT EXISTS idx_conv_chunks_timestamp ON conv_chunks(timestamp);

    -- FTS5 full-text search
    CREATE VIRTUAL TABLE IF NOT EXISTS conv_chunks_fts USING fts5(
      user_content,
      assistant_content,
      content=conv_chunks,
      content_rowid=rowid,
      tokenize='porter unicode61'
    );

    CREATE TRIGGER IF NOT EXISTS conv_chunks_ai AFTER INSERT ON conv_chunks BEGIN
      INSERT INTO conv_chunks_fts(rowid, user_content, assistant_content)
      VALUES (new.rowid, new.user_content, new.assistant_content);
    END;

    CREATE TRIGGER IF NOT EXISTS conv_chunks_ad AFTER DELETE ON conv_chunks BEGIN
      INSERT INTO conv_chunks_fts(conv_chunks_fts, rowid, user_content, assistant_content)
      VALUES ('delete', old.rowid, old.user_content, old.assistant_content);
    END;

    CREATE TRIGGER IF NOT EXISTS conv_chunks_au AFTER UPDATE ON conv_chunks BEGIN
      INSERT INTO conv_chunks_fts(conv_chunks_fts, rowid, user_content, assistant_content)
      VALUES ('delete', old.rowid, old.user_content, old.assistant_content);
      INSERT INTO conv_chunks_fts(rowid, user_content, assistant_content)
      VALUES (new.rowid, new.user_content, new.assistant_content);
    END;

    -- Stats (usage counters)
    CREATE TABLE IF NOT EXISTS stats (
      key   TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );

    -- Meta (embedding model tracking, migration state)
    CREATE TABLE IF NOT EXISTS meta (
      key   TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );

    -- Ignored projects
    CREATE TABLE IF NOT EXISTS ignored_projects (
      project    TEXT PRIMARY KEY,
      ignored_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
  `);
}

// --- Stats CRUD ---

export function getStat(db: Database.Database, key: string): string | null {
  const row = db.prepare('SELECT value FROM stats WHERE key = ?').get(key) as
    | { value: string }
    | undefined;
  return row?.value ?? null;
}

export function setStat(db: Database.Database, key: string, value: string): void {
  db.prepare('INSERT OR REPLACE INTO stats (key, value) VALUES (?, ?)').run(key, value);
}

/** One statement, so concurrent writers (daemon, hooks, other servers) never lose a count. */
export function incrementStat(db: Database.Database, key: string, delta = 1): void {
  db.prepare(
    `INSERT INTO stats (key, value) VALUES (?, ?)
     ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + CAST(excluded.value AS INTEGER)`,
  ).run(key, String(delta));
}

// --- Buffered usage counters ---

/** Queued counters reach the database in one short write this often. */
export const STATS_FLUSH_MS = 30_000;
/** A flush waits at most this long for another writer's lock, then drops its counts. */
const STATS_FLUSH_BUSY_MS = 100;

interface PendingStats {
  add: Map<string, number>;
  set: Map<string, string>;
}
const pendingStats = new Map<Database.Database, PendingStats>();
let statsTimer: ReturnType<typeof setInterval> | undefined;

/**
 * Queue counter increments and values for the next flush. Never touches the database, so a
 * search cannot wait on a lock held by a hook writing in another process.
 */
export function queueStats(
  db: Database.Database,
  add: Record<string, number>,
  set: Record<string, string> = {},
): void {
  let p = pendingStats.get(db);
  if (!p) {
    p = { add: new Map(), set: new Map() };
    pendingStats.set(db, p);
  }
  for (const [k, v] of Object.entries(add)) p.add.set(k, (p.add.get(k) ?? 0) + v);
  for (const [k, v] of Object.entries(set)) p.set.set(k, v);
  if (!statsTimer) {
    statsTimer = setInterval(flushAllStats, STATS_FLUSH_MS);
    statsTimer.unref();
  }
}

/**
 * Write one connection's queued counters in a single transaction. A flush that cannot get
 * the write lock within STATS_FLUSH_BUSY_MS, or fails otherwise, is dropped: these are
 * usage figures, not data. Returns false when counts were dropped.
 */
export function flushStats(db: Database.Database): boolean {
  const p = pendingStats.get(db);
  pendingStats.delete(db);
  if (pendingStats.size === 0 && statsTimer) {
    clearInterval(statsTimer);
    statsTimer = undefined;
  }
  if (!p) return true;
  if (!db.open) return false;
  const busy = db.pragma('busy_timeout', { simple: true }) as number;
  try {
    db.pragma(`busy_timeout = ${STATS_FLUSH_BUSY_MS}`);
    db.transaction(() => {
      for (const [k, v] of p.add) incrementStat(db, k, v);
      for (const [k, v] of p.set) setStat(db, k, v);
    }).immediate();
    return true;
  } catch {
    return false;
  } finally {
    if (db.open) db.pragma(`busy_timeout = ${busy}`);
  }
}

/** A stat as readers should see it: queued increments added, a queued value preferred. */
export function readStat(db: Database.Database, key: string): string | null {
  const p = pendingStats.get(db);
  const queued = p?.set.get(key);
  if (queued !== undefined) return queued;
  const stored = getStat(db, key);
  const add = p?.add.get(key);
  return add === undefined ? stored : String((stored ? parseInt(stored, 10) : 0) + add);
}

/** Flush every connection's queued counters (the timer, and shutdown). */
export function flushAllStats(): void {
  for (const db of [...pendingStats.keys()]) flushStats(db);
}

// --- Vector CRUD ---

export function insertVector(
  db: Database.Database,
  chunkId: string,
  embedding: Float32Array,
  suffix = '_text',
): void {
  insertVectorsBatchToTable(db, [{ chunkId, embedding }], suffix);
}

export function insertVectorsBatch(
  db: Database.Database,
  items: Array<{ chunkId: string; embedding: Float32Array }>,
  suffix = '_text',
): void {
  insertVectorsBatchToTable(db, items, suffix);
}

/**
 * Publish freshly computed vectors, safely against concurrent writers.
 *
 * Inference runs outside any transaction, so by the time a batch is ready a hook may
 * have rewritten or deleted some of its chunks. In one short transaction each item is
 * rechecked: the chunk must still exist, be undeleted and carry the hash it was read
 * with, and must not already have a vector in this generation. Stale or duplicate items
 * are discarded (the next pass re-reads them), so no wrong vector is attached and no
 * vec0 row leaks. Returns the number of vectors actually committed.
 */
export function publishVectors(
  db: Database.Database,
  items: Array<{ chunkId: string; hash: string; embedding: Float32Array }>,
  suffix = '_text',
): number {
  if (items.length === 0) return 0;
  const current = db.prepare('SELECT hash FROM conv_chunks WHERE id = ? AND deleted_at IS NULL');
  const mapped = db.prepare(`SELECT 1 FROM conv_vec_map${suffix} WHERE chunk_id = ?`);
  const insVec = db.prepare(`INSERT INTO conv_vec${suffix} (embedding) VALUES (?)`);
  const insMap = db.prepare(
    `INSERT INTO conv_vec_map${suffix} (vec_rowid, chunk_id) VALUES (?, ?)`,
  );
  return db
    .transaction(() => {
      let published = 0;
      for (const { chunkId, hash, embedding } of items) {
        const row = current.get(chunkId) as { hash: string } | undefined;
        if (!row || row.hash !== hash || mapped.get(chunkId)) continue;
        const vecRowid = insVec.run(Buffer.from(embedding.buffer)).lastInsertRowid;
        insMap.run(vecRowid, chunkId);
        published++;
      }
      return published;
    })
    .immediate();
}

// --- Deletion: the one path every caller uses ---
//
// conv_vec_map* rows cascade away with their chunk, but the vec0 rows they point to do
// not: a plain chunk DELETE leaves orphan vectors behind. Every chunk or session delete
// therefore goes through deleteChunksWhere / deleteSessionsWhere, which remove vectors
// from every generation that exists (text, code, next), enabled or not.

const vecLoaded = new WeakMap<Database.Database, boolean>();

/** True when this connection has sqlite-vec loaded (vec0 tables are usable). */
export function hasVec(db: Database.Database): boolean {
  let v = vecLoaded.get(db);
  if (v === undefined) {
    try {
      db.prepare('SELECT vec_version()').get();
      v = true;
    } catch {
      v = false;
    }
    vecLoaded.set(db, v);
  }
  return v;
}

/** Every vector map table that exists, paired with its vec0 table. */
export function vectorTables(db: Database.Database): Array<{ map: string; vec: string }> {
  const names = db
    .prepare(
      "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'conv\\_vec\\_map%' ESCAPE '\\'",
    )
    .pluck()
    .all() as string[];
  return names.map((map) => ({ map, vec: 'conv_vec' + map.slice('conv_vec_map'.length) }));
}

function inTransaction<T>(db: Database.Database, fn: () => T): T {
  return db.inTransaction ? fn() : db.transaction(fn)();
}

const ID_BATCH = 500;

/**
 * Remove the vectors of these chunks from every generation. In a process without
 * sqlite-vec the vec0 rowids are queued in pending_vec_cleanup (drained by the daemon)
 * instead of failing. Returns the number of map rows removed.
 */
export function deleteVectorsForChunks(db: Database.Database, chunkIds: string[]): number {
  if (chunkIds.length === 0) return 0;
  const withVec = hasVec(db);
  let removed = 0;
  inTransaction(db, () => {
    const queue = withVec
      ? null
      : db.prepare(
          'INSERT OR IGNORE INTO pending_vec_cleanup (vec_table, vec_rowid) VALUES (?, ?)',
        );
    for (const { map, vec } of vectorTables(db)) {
      const del = withVec ? db.prepare(`DELETE FROM ${vec} WHERE rowid = ?`) : null;
      for (let i = 0; i < chunkIds.length; i += ID_BATCH) {
        const batch = chunkIds.slice(i, i + ID_BATCH);
        const ph = batch.map(() => '?').join(',');
        const rowids = db
          .prepare(`SELECT vec_rowid FROM ${map} WHERE chunk_id IN (${ph})`)
          .pluck()
          .all(...batch) as number[];
        if (rowids.length === 0) continue;
        for (const rowid of rowids) {
          if (del) del.run(rowid);
          else queue!.run(vec, rowid);
        }
        db.prepare(`DELETE FROM ${map} WHERE chunk_id IN (${ph})`).run(...batch);
        removed += rowids.length;
      }
    }
  });
  return removed;
}

/** Delete chunks matching a WHERE clause (internal SQL only), with all their vectors. */
export function deleteChunksWhere(
  db: Database.Database,
  where: string,
  params: unknown[] = [],
): number {
  return inTransaction(db, () => {
    const ids = db
      .prepare(`SELECT id FROM conv_chunks WHERE ${where}`)
      .pluck()
      .all(...params) as string[];
    deleteVectorsForChunks(db, ids);
    return db.prepare(`DELETE FROM conv_chunks WHERE ${where}`).run(...params).changes;
  });
}

/** Delete sessions matching a WHERE clause, their chunks and all those chunks' vectors. */
export function deleteSessionsWhere(
  db: Database.Database,
  where: string,
  params: unknown[] = [],
): { sessions: number; chunks: number } {
  return inTransaction(db, () => {
    const chunks = deleteChunksWhere(
      db,
      `session_id IN (SELECT id FROM conv_sessions WHERE ${where})`,
      params,
    );
    const sessions = db.prepare(`DELETE FROM conv_sessions WHERE ${where}`).run(...params).changes;
    return { sessions, chunks };
  });
}

/** Apply queued vec0 deletions from processes that could not load sqlite-vec. */
export function drainPendingVecCleanup(db: Database.Database): number {
  if (!hasVec(db)) return 0;
  return inTransaction(db, () => {
    const rows = db.prepare('SELECT vec_table, vec_rowid FROM pending_vec_cleanup').all() as Array<{
      vec_table: string;
      vec_rowid: number;
    }>;
    const existing = new Set(
      db.prepare("SELECT name FROM sqlite_master WHERE type = 'table'").pluck().all() as string[],
    );
    for (const r of rows) {
      if (existing.has(r.vec_table)) {
        db.prepare(`DELETE FROM ${r.vec_table} WHERE rowid = ?`).run(r.vec_rowid);
      }
    }
    db.prepare('DELETE FROM pending_vec_cleanup').run();
    return rows.length;
  });
}

/**
 * Minimum content length for embedding. Chunks shorter than this produce
 * noisy vectors that pollute search results at scale.
 */
export const MIN_EMBED_CONTENT_LENGTH = 50;

export function getChunksWithoutEmbeddings(
  db: Database.Database,
  limit = 100,
  suffix = '_text',
): Array<{ id: string; content: string; hash: string }> {
  return db
    .prepare(
      `SELECT c.id, (c.user_content || ' ' || c.assistant_content) AS content, c.hash
       FROM conv_chunks c
       LEFT JOIN conv_vec_map${suffix} m ON c.id = m.chunk_id
       WHERE m.chunk_id IS NULL
         AND c.deleted_at IS NULL
         AND LENGTH(c.user_content) + LENGTH(c.assistant_content) >= ?
       ORDER BY c.timestamp DESC
       LIMIT ?`,
    )
    .all(MIN_EMBED_CONTENT_LENGTH, limit) as Array<{ id: string; content: string; hash: string }>;
}

export function countEligibleChunks(db: Database.Database): number {
  const row = db
    .prepare(
      'SELECT COUNT(*) AS cnt FROM conv_chunks WHERE deleted_at IS NULL AND LENGTH(user_content) + LENGTH(assistant_content) >= ?',
    )
    .get(MIN_EMBED_CONTENT_LENGTH) as { cnt: number };
  return row.cnt;
}

export function countEmbeddedChunks(db: Database.Database, suffix = '_text'): number {
  const row = db.prepare(`SELECT COUNT(*) AS cnt FROM conv_vec_map${suffix}`).get() as {
    cnt: number;
  };
  return row.cnt;
}

// --- Meta CRUD ---

export function getMeta(db: Database.Database, key: string): string | null {
  const row = db.prepare('SELECT value FROM meta WHERE key = ?').get(key) as
    | { value: string }
    | undefined;
  return row?.value ?? null;
}

export function setMeta(db: Database.Database, key: string, value: string): void {
  db.prepare('INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)').run(key, value);
}

export function deleteMeta(db: Database.Database, key: string): void {
  db.prepare('DELETE FROM meta WHERE key = ?').run(key);
}

export function deleteMetaByPrefix(db: Database.Database, prefix: string): void {
  db.prepare('DELETE FROM meta WHERE key LIKE ?').run(`${prefix}%`);
}

// --- Parameterized vec table creation ---

/**
 * Read the actual dimensions from a vec0 virtual table's schema in sqlite_master.
 * Returns null if the table doesn't exist or the schema can't be parsed.
 */
export function getVecTableDimensions(db: Database.Database, suffix = '_text'): number | null {
  const row = db
    .prepare(`SELECT sql FROM sqlite_master WHERE type='table' AND name=?`)
    .get(`conv_vec${suffix}`) as { sql: string } | undefined;
  if (!row?.sql) return null;
  const match = row.sql.match(/float\[(\d+)\]/);
  return match ? parseInt(match[1], 10) : null;
}

/**
 * Recreate primary vec tables with new dimensions.
 * Drops existing tables and all associated vectors.
 * Used when the embedding model changes dimensions (e.g. MiniLM 384d → Ollama 768d).
 */
export function recreateVecTables(
  db: Database.Database,
  dimensions: number,
  suffix = '_text',
): void {
  // Drop in dependency order — vec_map first (references chunks_vec via rowid)
  db.exec(`DROP TABLE IF EXISTS conv_vec_map${suffix}`);
  db.exec(`DROP TABLE IF EXISTS conv_vec${suffix}`);
  createVecTables(db, dimensions, suffix);
}

export function createVecTables(db: Database.Database, dimensions: number, suffix = ''): void {
  db.exec(`
    CREATE VIRTUAL TABLE IF NOT EXISTS conv_vec${suffix} USING vec0(
      embedding float[${dimensions}]
    );

    CREATE TABLE IF NOT EXISTS conv_vec_map${suffix} (
      vec_rowid INTEGER PRIMARY KEY,
      chunk_id TEXT NOT NULL UNIQUE REFERENCES conv_chunks(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_conv_vec_map_chunk${suffix} ON conv_vec_map${suffix}(chunk_id);
  `);
}

/** Insert vectors without a content check (tests, tools). Never leaks: mapped chunks are skipped. */
export function insertVectorsBatchToTable(
  db: Database.Database,
  items: Array<{ chunkId: string; embedding: Float32Array }>,
  suffix = '',
): void {
  const mapped = db.prepare(`SELECT 1 FROM conv_vec_map${suffix} WHERE chunk_id = ?`);
  const insertVec = db.prepare(`INSERT INTO conv_vec${suffix} (embedding) VALUES (?)`);
  const insertMap = db.prepare(
    `INSERT INTO conv_vec_map${suffix} (vec_rowid, chunk_id) VALUES (?, ?)`,
  );

  const transaction = db.transaction(() => {
    for (const { chunkId, embedding } of items) {
      if (mapped.get(chunkId)) continue;
      const vecRowid = insertVec.run(Buffer.from(embedding.buffer)).lastInsertRowid;
      insertMap.run(vecRowid, chunkId);
    }
  });

  transaction();
}

// --- Ignored projects ---

export function isProjectIgnored(db: Database.Database, project: string): boolean {
  const row = db.prepare('SELECT 1 FROM ignored_projects WHERE project = ?').get(project);
  return !!row;
}

export function ignoreProject(db: Database.Database, project: string): void {
  db.prepare('INSERT OR IGNORE INTO ignored_projects (project) VALUES (?)').run(project);
}

export function unignoreProject(db: Database.Database, project: string): void {
  db.prepare('DELETE FROM ignored_projects WHERE project = ?').run(project);
}

export function getIgnoredProjects(
  db: Database.Database,
): Array<{ project: string; ignoredAt: string }> {
  return db
    .prepare(
      'SELECT project, ignored_at AS ignoredAt FROM ignored_projects ORDER BY ignored_at DESC',
    )
    .all() as Array<{
    project: string;
    ignoredAt: string;
  }>;
}

export interface PurgeResult {
  sessionsPurged: number;
  chunksPurged: number;
}

export function purgeProjectData(db: Database.Database, project: string): PurgeResult {
  const { sessions, chunks } = deleteSessionsWhere(db, 'project = ?', [project]);
  return { sessionsPurged: sessions, chunksPurged: chunks };
}

export function closeDatabase(db: Database.Database): void {
  flushStats(db);
  db.close();
}
