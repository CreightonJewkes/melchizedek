import { describe, it, expect, afterEach } from 'vitest';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { fork } from 'child_process';
import Database from 'better-sqlite3';
import {
  openDatabase,
  initSchema,
  loadSqliteVec,
  SCHEMA_STEPS,
  SCHEMA_VERSION,
  SchemaVersionError,
  type SchemaStep,
} from '../src/db.js';
import {
  createV2Database,
  seedSyntheticCorpus,
  archiveCounts,
  tempDbPath,
} from './helpers/fixtures.js';

const version = (db: Database.Database): number =>
  db.pragma('user_version', { simple: true }) as number;

function v2Fixture(): { dbPath: string; cleanup: () => void; counts: Record<string, number> } {
  const tmp = tempDbPath();
  createV2Database(tmp.dbPath);
  const db = new Database(tmp.dbPath);
  seedSyntheticCorpus(db);
  loadSqliteVec(db);
  const vecRowid = db
    .prepare('INSERT INTO conv_vec_text (embedding) VALUES (?)')
    .run(Buffer.from(new Float32Array(384).fill(0.05).buffer)).lastInsertRowid;
  db.prepare('INSERT INTO conv_vec_map_text (vec_rowid, chunk_id) VALUES (?, ?)').run(
    vecRowid,
    'sess-archive:0',
  );
  const counts = archiveCounts(db);
  db.close();
  return { ...tmp, counts };
}

describe('schema upgrades (L1)', () => {
  let cleanup = (): void => {};
  afterEach(() => cleanup());

  it('upgrades a 1.0.2 database in place and keeps every archived row', () => {
    const fx = v2Fixture();
    cleanup = fx.cleanup;

    const { db, schemaReady } = openDatabase(fx.dbPath);
    expect(schemaReady).toBe(true);
    expect(version(db)).toBe(SCHEMA_VERSION);
    expect(archiveCounts(db)).toEqual(fx.counts);

    const archive = db
      .prepare("SELECT jsonl_path FROM conv_sessions WHERE id = 'sess-archive'")
      .get();
    expect(archive).toEqual({ jsonl_path: '/nonexistent/archive/sess-archive.jsonl' });
    const fts = db
      .prepare("SELECT COUNT(*) AS n FROM conv_chunks_fts WHERE conv_chunks_fts MATCH 'lunar'")
      .get() as { n: number };
    expect(fts.n).toBe(1);
    const mapped = db.prepare('SELECT chunk_id FROM conv_vec_map_text').all();
    expect(mapped).toEqual([{ chunk_id: 'sess-archive:0' }]);
    db.close();
  });

  it('rewrites space-separated chunk timestamps in ISO-8601 form', () => {
    const fx = v2Fixture();
    cleanup = fx.cleanup;
    const { db } = openDatabase(fx.dbPath);
    const ts = (id: string): string =>
      (
        db.prepare('SELECT timestamp FROM conv_chunks WHERE id = ?').get(id) as {
          timestamp: string;
        }
      ).timestamp;
    expect(ts('sess-live:1')).toBe('2026-03-01T09:30:00.000Z');
    expect(ts('__manual_memories__:0')).toBe('2026-02-10T12:00:00.000Z');
    expect(ts('sess-live:0')).toBe('2026-03-01T09:00:00.000Z');
    const spaced = db
      .prepare("SELECT COUNT(*) AS n FROM conv_chunks WHERE timestamp LIKE '____-__-__ %'")
      .get() as { n: number };
    expect(spaced.n).toBe(0);
    db.close();
  });

  for (const step of SCHEMA_STEPS) {
    for (const where of ['ddl', 'run'] as const) {
      it(`a failure in step ${step.version} (${where}) leaves the database at the old version`, () => {
        const fx = v2Fixture();
        cleanup = fx.cleanup;
        const db = new Database(fx.dbPath);
        db.pragma('foreign_keys = ON');
        const vec = loadSqliteVec(db);
        const failing: SchemaStep[] = SCHEMA_STEPS.map((s) =>
          s.version !== step.version
            ? s
            : where === 'ddl'
              ? {
                  ...s,
                  ddl: () => {
                    throw new Error('injected');
                  },
                }
              : {
                  ...s,
                  run: (d) => {
                    s.run(d);
                    throw new Error('injected');
                  },
                },
        );
        expect(() => initSchema(db, vec, failing)).toThrow('injected');
        expect(version(db)).toBe(step.version - 1);
        expect(archiveCounts(db)).toEqual(fx.counts);
        db.close();

        // A later normal open completes the upgrade.
        const reopened = openDatabase(fx.dbPath);
        expect(version(reopened.db)).toBe(SCHEMA_VERSION);
        expect(archiveCounts(reopened.db)).toEqual(fx.counts);
        reopened.db.close();
      });
    }
  }

  it('refuses a database from newer code without writing to it', () => {
    const fx = v2Fixture();
    cleanup = fx.cleanup;
    const raw = new Database(fx.dbPath);
    raw.pragma('journal_mode = DELETE');
    raw.pragma(`user_version = ${SCHEMA_VERSION + 1}`);
    raw.close();
    const before = crypto.createHash('sha256').update(fs.readFileSync(fx.dbPath)).digest('hex');

    expect(() => openDatabase(fx.dbPath)).toThrow(SchemaVersionError);

    const after = crypto.createHash('sha256').update(fs.readFileSync(fx.dbPath)).digest('hex');
    expect(after).toBe(before);
    expect(fs.existsSync(fx.dbPath + '-wal')).toBe(false);
  });

  it('defers the upgrade, without error, while another writer holds the lock', () => {
    const fx = v2Fixture();
    cleanup = fx.cleanup;
    const holder = new Database(fx.dbPath);
    holder.prepare('BEGIN IMMEDIATE').run();

    const db = new Database(fx.dbPath);
    db.pragma('busy_timeout = 50');
    expect(initSchema(db, loadSqliteVec(db))).toBe(false);
    expect(version(db)).toBe(2);

    holder.prepare('ROLLBACK').run();
    holder.close();
    expect(initSchema(db, true)).toBe(true);
    expect(version(db)).toBe(SCHEMA_VERSION);
    expect(archiveCounts(db)).toEqual(fx.counts);
    db.close();
  });

  it('survives several processes opening the same old database at once', async () => {
    const fx = v2Fixture();
    cleanup = fx.cleanup;
    const script = path.join(path.dirname(fx.dbPath), 'open.mts');
    const dbModule = path.join(import.meta.dirname, '..', 'src', 'db.ts');
    fs.writeFileSync(
      script,
      `import { createRequire } from 'module';
       globalThis.require = createRequire(import.meta.url);
       const { openDatabase } = await import(${JSON.stringify(dbModule)});
       const { db, schemaReady } = openDatabase(process.argv[2]);
       process.send({ schemaReady, version: db.pragma('user_version', { simple: true }) });
       db.close();`,
    );
    const runs = Array.from(
      { length: 4 },
      () =>
        new Promise<{ schemaReady: boolean; version: number; code: number | null }>((resolve) => {
          const child = fork(script, [fx.dbPath], { execArgv: ['--import', 'tsx'] });
          let msg = { schemaReady: false, version: -1 };
          child.on('message', (m) => (msg = m as typeof msg));
          child.on('exit', (code) => resolve({ ...msg, code }));
        }),
    );
    const results = await Promise.all(runs);
    for (const r of results) expect(r.code).toBe(0);
    expect(results.some((r) => r.schemaReady && r.version === SCHEMA_VERSION)).toBe(true);

    const db = new Database(fx.dbPath);
    expect(version(db)).toBe(SCHEMA_VERSION);
    expect(archiveCounts(db)).toEqual(fx.counts);
    db.close();
  }, 30_000);

  it('keeps legacy version-1 tables untouched and creates the current schema beside them', () => {
    const tmp = tempDbPath();
    cleanup = tmp.cleanup;
    const raw = new Database(tmp.dbPath);
    raw.exec(`CREATE TABLE sessions (id TEXT PRIMARY KEY, body TEXT);
              INSERT INTO sessions VALUES ('old-1', 'irreplaceable');
              PRAGMA user_version = 1;`);
    raw.close();

    const { db } = openDatabase(tmp.dbPath);
    expect(version(db)).toBe(SCHEMA_VERSION);
    expect(db.prepare('SELECT body FROM sessions').get()).toEqual({ body: 'irreplaceable' });
    expect(db.prepare('SELECT COUNT(*) AS n FROM conv_chunks').get()).toEqual({ n: 0 });
    db.close();
  });
});
