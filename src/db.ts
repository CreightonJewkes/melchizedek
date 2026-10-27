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

export function incrementStat(db: Database.Database, key: string, delta = 1): void {
  const current = getStat(db, key);
  const newValue = (current ? parseInt(current, 10) : 0) + delta;
  setStat(db, key, String(newValue));
}

// --- Vector CRUD ---

export function insertVector(
  db: Database.Database,
  chunkId: string,
  embedding: Float32Array,
  suffix = '_text',
): void {
  const vecRowid = db
    .prepare(`INSERT INTO conv_vec${suffix} (embedding) VALUES (?)`)
    .run(Buffer.from(embedding.buffer)).lastInsertRowid;
  db.prepare(`INSERT OR IGNORE INTO conv_vec_map${suffix} (vec_rowid, chunk_id) VALUES (?, ?)`).run(
    vecRowid,
    chunkId,
  );
}

export function insertVectorsBatch(
  db: Database.Database,
  items: Array<{ chunkId: string; embedding: Float32Array }>,
  suffix = '_text',
): void {
  insertVectorsBatchToTable(db, items, suffix);
}

// --- Vector deletion ---

export function deleteVectorsForChunk(
  db: Database.Database,
  chunkId: string,
  suffix = '_text',
): void {
  const map = db
    .prepare(`SELECT vec_rowid FROM conv_vec_map${suffix} WHERE chunk_id = ?`)
    .get(chunkId) as { vec_rowid: number } | undefined;
  if (map) {
    db.prepare(`DELETE FROM conv_vec${suffix} WHERE rowid = ?`).run(map.vec_rowid);
    db.prepare(`DELETE FROM conv_vec_map${suffix} WHERE chunk_id = ?`).run(chunkId);
  }
}

/**
 * Minimum content length for embedding. Chunks shorter than this produce
 * noisy vectors that pollute search results at scale.
 */
const MIN_EMBED_CONTENT_LENGTH = 50;

export function getChunksWithoutEmbeddings(
  db: Database.Database,
  limit = 100,
  suffix = '_text',
): Array<{ id: string; content: string }> {
  return db
    .prepare(
      `SELECT c.id, (c.user_content || ' ' || c.assistant_content) AS content
       FROM conv_chunks c
       LEFT JOIN conv_vec_map${suffix} m ON c.id = m.chunk_id
       WHERE m.chunk_id IS NULL
         AND c.deleted_at IS NULL
         AND LENGTH(c.user_content) + LENGTH(c.assistant_content) >= ?
       ORDER BY c.timestamp DESC
       LIMIT ?`,
    )
    .all(MIN_EMBED_CONTENT_LENGTH, limit) as Array<{ id: string; content: string }>;
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

export function insertVectorsBatchToTable(
  db: Database.Database,
  items: Array<{ chunkId: string; embedding: Float32Array }>,
  suffix = '',
): void {
  const insertVec = db.prepare(`INSERT INTO conv_vec${suffix} (embedding) VALUES (?)`);
  const insertMap = db.prepare(
    `INSERT OR IGNORE INTO conv_vec_map${suffix} (vec_rowid, chunk_id) VALUES (?, ?)`,
  );

  const transaction = db.transaction(() => {
    for (const { chunkId, embedding } of items) {
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
  // Count chunks before deletion
  const chunkCount = (
    db
      .prepare(
        'SELECT COUNT(*) AS cnt FROM conv_chunks WHERE session_id IN (SELECT id FROM conv_sessions WHERE project = ?)',
      )
      .get(project) as { cnt: number }
  ).cnt;

  // Delete sessions — chunks cascade via FK ON DELETE CASCADE
  const result = db.prepare('DELETE FROM conv_sessions WHERE project = ?').run(project);

  return {
    sessionsPurged: result.changes,
    chunksPurged: chunkCount,
  };
}

export function closeDatabase(db: Database.Database): void {
  db.close();
}
