import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { EventEmitter } from 'events';
import { EmbedOrchestrator, EMBED_WORKER_THREADS } from '../src/embed-orchestrator.js';
import {
  openMemoryDatabase,
  hasPendingEmbeddings,
  oldestPendingEmbedding,
  setMeta,
} from '../src/db.js';
import type { DatabaseType } from '../src/db.js';
import type { EmbedJobConfig, WorkerOutMessage, WorkerStartMessage } from '../src/models.js';

/** A worker stand-in. With `stuck`, SIGTERM is ignored, as in a native inference call. */
class FakeWorker extends EventEmitter {
  pid = 0;
  stuck = false;
  signals: string[] = [];
  sent: unknown[] = [];
  send(msg: unknown): boolean {
    this.sent.push(msg);
    return true;
  }
  kill(signal = 'SIGTERM'): boolean {
    this.signals.push(signal);
    if (signal === 'SIGTERM' && this.stuck) return true;
    setImmediate(() => this.emit('exit', null, signal));
    return true;
  }
  say(msg: WorkerOutMessage): void {
    this.emit('message', msg);
  }
}

const workers: FakeWorker[] = [];
let stuckNext = false;
vi.mock('child_process', () => ({
  fork: vi.fn(() => {
    const w = new FakeWorker();
    w.stuck = stuckNext;
    workers.push(w);
    return w;
  }),
}));

const job: EmbedJobConfig = {
  suffix: '_text',
  embedderType: 'text',
  config: { embeddingBackend: 'auto', embeddingModel: null, ollamaBaseUrl: 'http://x' },
};
const allJobs = {
  textEnabled: true,
  codeEnabled: false,
  config: {
    embeddingTextBackend: 'auto',
    embeddingTextModel: null,
    embeddingCodeBackend: 'auto',
    embeddingCodeModel: null,
    ollamaBaseUrl: 'http://x',
  },
};
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const progress = (embedded: number): WorkerOutMessage => ({
  type: 'progress',
  embedded,
  total: 100,
});

describe('continuous embedding (B3)', () => {
  let db: DatabaseType;
  let orch: EmbedOrchestrator;

  beforeEach(() => {
    workers.length = 0;
    stuckNext = false;
    db = openMemoryDatabase().db;
    orch = new EmbedOrchestrator('/tmp/x.db', db, {
      workerPath: '/fake/worker.js',
      noProgressMs: 60,
      killGraceMs: 40,
    });
  });
  afterEach(() => db.close());

  it('stops a worker that makes no progress: SIGTERM, SIGKILL after the grace, idle after exit', async () => {
    stuckNext = true;
    const run = orch.runJob(job);
    await sleep(80);
    expect(workers[0].signals).toEqual(['SIGTERM']);
    expect(orch.getStatus().active).toBe(true); // the child has not exited yet

    const r = await run;
    expect(workers[0].signals).toEqual(['SIGTERM', 'SIGKILL']);
    expect(r.error).toMatch(/no progress/);
    expect(orch.getStatus().active).toBe(false);
    expect(orch.getLastError()?.message).toMatch(/no progress/);
  });

  it('keeps a slow job alive as long as batches keep finishing', async () => {
    const run = orch.runJob(job);
    for (let i = 1; i <= 6; i++) {
      await sleep(30); // 180 ms in all, three times the no-progress limit
      workers[0].say(progress(i * 10));
    }
    workers[0].say({ type: 'done', embedded: 60, durationMs: 180 });
    const r = await run;
    expect(r).toEqual({ embedded: 60, error: undefined });
    expect(workers[0].signals).toEqual([]);
  });

  it('a second trigger while a job runs starts no second worker', async () => {
    const first = orch.runAllJobs(allJobs);
    await sleep(5);
    expect(orch.isBusy()).toBe(true);
    expect(await orch.runAllJobs(allJobs)).toBe(false); // skipped, not queued
    expect(workers).toHaveLength(1);
    workers[0].say({ type: 'done', embedded: 1, durationMs: 1 });
    expect(await first).toBe(true);
    expect(orch.isBusy()).toBe(false);
  });

  it('maintenance waits its turn: it never runs during an embed job', async () => {
    const first = orch.runAllJobs(allJobs);
    await sleep(5);
    let ran = false;
    expect(
      await orch.runExclusive(async () => {
        ran = true;
      }),
    ).toBe(false);
    expect(ran).toBe(false);
    workers[0].say({ type: 'done', embedded: 1, durationMs: 1 });
    await first;
    expect(
      await orch.runExclusive(async () => {
        ran = true;
      }),
    ).toBe(true);
    expect(ran).toBe(true);
  });

  it('a failing task neither rejects nor holds the flight; its error is kept', async () => {
    // The daemon runs upkeep from a timer: a rejection here would end the process.
    expect(
      await orch.runExclusive(async () => {
        throw new Error('database is locked');
      }),
    ).toBe(true);
    expect(orch.getLastError()?.message).toBe('database is locked');
    expect(orch.isBusy()).toBe(false);
    let ran = false;
    await orch.runExclusive(async () => {
      ran = true;
    });
    expect(ran).toBe(true);
  });

  it('records a crash as the last error, and the next run starts cleanly', async () => {
    const run = orch.runJob(job);
    await sleep(5);
    workers[0].emit('exit', 1, null);
    expect((await run).error).toMatch(/code 1/);
    expect(orch.getLastError()?.message).toMatch(/code 1/);

    const again = orch.runJob(job);
    await sleep(5);
    expect(workers).toHaveLength(2);
    workers[1].say({ type: 'done', embedded: 3, durationMs: 1 });
    expect((await again).error).toBeUndefined();
    expect(orch.getLastError()).toBeNull();
  });

  it('asks the worker to cap its inference threads', async () => {
    const run = orch.runJob(job);
    await sleep(5);
    expect((workers[0].sent[0] as WorkerStartMessage).threads).toBe(EMBED_WORKER_THREADS);
    workers[0].say({ type: 'done', embedded: 0, durationMs: 1 });
    await run;
  });
});

describe('pending work for the embedding timer (B3)', () => {
  let db: DatabaseType;
  beforeEach(() => {
    db = openMemoryDatabase().db;
    db.prepare(
      "INSERT INTO conv_sessions (id, project, jsonl_path, started_at) VALUES ('s', '/p', '/t', '2026-01-01')",
    ).run();
  });
  afterEach(() => db.close());

  const add = (id: string, text: string, indexedAt: string | null) =>
    db
      .prepare(
        `INSERT INTO conv_chunks (id, session_id, idx, user_content, assistant_content, hash, timestamp, indexed_at)
         VALUES (?, 's', 0, ?, '', ?, '2026-01-01T00:00:00Z', ?)`,
      )
      .run(id, text, id, indexedAt);

  it('finds eligible chunks without a vector, and the longest wait among them', () => {
    expect(hasPendingEmbeddings(db, '_text')).toBe(false);
    add('short', 'too short to embed', '2026-01-01T00:00:00Z');
    expect(hasPendingEmbeddings(db, '_text')).toBe(false);
    add('a', 'x'.repeat(60), '2026-01-03T00:00:00Z');
    add('b', 'y'.repeat(60), '2026-01-02T00:00:00Z');
    expect(hasPendingEmbeddings(db, '_text')).toBe(true);
    expect(oldestPendingEmbedding(db, '_text')).toBe('2026-01-02T00:00:00Z');
    expect(hasPendingEmbeddings(db, '_code')).toBe(false); // no code tables
  });

  it('a migration in progress counts as work', () => {
    setMeta(db, 'migration_target_model_text', 'gte-small');
    expect(hasPendingEmbeddings(db, '_text')).toBe(true);
  });
});
