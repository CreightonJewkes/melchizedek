/**
 * EmbedOrchestrator — spawns and manages child process embed workers.
 * Workers run sequentially (text first, then code).
 * Progress is tracked in the meta table for m9k_info visibility.
 */

import { fork, type ChildProcess } from 'child_process';
import path from 'path';
import { setMeta, deleteMeta } from './db.js';
import { logger } from './logger.js';
import type { DatabaseType } from './db.js';
import type {
  EmbedJobConfig,
  EmbedJobStatus,
  WorkerStartMessage,
  WorkerOutMessage,
} from './models.js';
import { EMBED_NO_PROGRESS_MS, EMBED_KILL_GRACE_MS } from './constants.js';

const P = 'embed-orch';

/** ONNX inference threads per embed worker: background work must not lag the machine. */
export const EMBED_WORKER_THREADS = 2;

export class EmbedOrchestrator {
  private dbPath: string;
  private db: DatabaseType;
  private workerPath: string;
  private child: ChildProcess | null = null;
  private status: EmbedJobStatus = {
    active: false,
    suffix: null,
    embedded: 0,
    total: 0,
    pid: null,
    rssMB: null,
    heapUsedMB: null,
  };
  private statusListeners = new Set<(status: EmbedJobStatus) => void>();
  private busy = false;
  private lastError: { message: string; at: string } | null = null;
  private readonly noProgressMs: number;
  private readonly killGraceMs: number;

  constructor(
    dbPath: string,
    db: DatabaseType,
    options?: { workerPath?: string; noProgressMs?: number; killGraceMs?: number },
  ) {
    this.dbPath = dbPath;
    this.db = db;
    this.workerPath = options?.workerPath ?? path.join(import.meta.dirname, 'embed-worker.js');
    this.noProgressMs = options?.noProgressMs ?? EMBED_NO_PROGRESS_MS;
    this.killGraceMs = options?.killGraceMs ?? EMBED_KILL_GRACE_MS;

    // Clean up stale progress from a previous crash
    this.cleanStaleProgress();
  }

  getStatus(): EmbedJobStatus {
    return { ...this.status };
  }

  /** Subscribe to status changes (one listener per connection); returns the unsubscribe. */
  onStatusChange(cb: (status: EmbedJobStatus) => void): () => void {
    this.statusListeners.add(cb);
    return () => this.statusListeners.delete(cb);
  }

  /** True while an embed job or maintenance task holds the single flight. */
  isBusy(): boolean {
    return this.busy;
  }

  /** The most recent job's error, if it failed (cleared by the next successful job). */
  getLastError(): { message: string; at: string } | null {
    return this.lastError ? { ...this.lastError } : null;
  }

  /**
   * Single flight for everything that writes vectors: embed jobs and maintenance (the
   * orphan sweep must never run while a worker publishes). A call while another runs
   * returns false at once rather than queueing; the periodic triggers simply retry.
   * A failing task never rejects (the daemon calls this from timers, where a rejection
   * would end the process): its error is logged and kept as the last error.
   */
  async runExclusive(fn: () => Promise<void>): Promise<boolean> {
    if (this.busy) return false;
    this.busy = true;
    try {
      await fn();
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      logger.warn(P, `Background task failed: ${message}`);
      this.lastError = { message, at: new Date().toISOString() };
    } finally {
      this.busy = false;
    }
    return true;
  }

  async runJob(job: EmbedJobConfig): Promise<{ embedded: number; error?: string }> {
    return new Promise((resolve) => {
      const batchSize = job.batchSize ?? 50;

      this.status = {
        active: true,
        suffix: job.suffix,
        embedded: 0,
        total: 0,
        pid: null,
        rssMB: null,
        heapUsedMB: null,
      };
      this.notifyStatusChange();

      let child: ChildProcess;
      try {
        child = fork(this.workerPath, [], {
          stdio: ['pipe', 'pipe', 'inherit', 'ipc'],
          execArgv: ['--max-old-space-size=2048'],
        });
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        this.resetStatus();
        resolve({ embedded: 0, error: message });
        return;
      }

      this.child = child;
      this.status.pid = child.pid ?? null;
      this.notifyStatusChange();

      let embedded = 0;
      let resolved = false;
      // Set once the worker is being stopped; the job ends when the child has exited.
      let stopping: string | null = null;
      let progressTimer: ReturnType<typeof setTimeout> | undefined;
      let killTimer: ReturnType<typeof setTimeout> | undefined;

      const finish = (error?: string) => {
        if (resolved) return;
        resolved = true;
        clearTimeout(progressTimer);
        clearTimeout(killTimer);
        this.cleanProgressMeta(job.suffix);
        this.lastError = error ? { message: error, at: new Date().toISOString() } : null;
        this.resetStatus();
        resolve({ embedded, error });
      };

      // Stop the worker: SIGTERM, then SIGKILL after the grace period (a worker blocked in
      // native inference cannot run its SIGTERM handler). finish() waits for 'exit'.
      const stop = (reason: string) => {
        if (resolved || stopping) return;
        stopping = reason;
        logger.warn(P, `Stopping embed worker: ${reason}`);
        child.kill('SIGTERM');
        killTimer = setTimeout(() => {
          if (!resolved) child.kill('SIGKILL');
        }, this.killGraceMs);
      };

      // No total time cap: a slow job may run as long as batches keep finishing.
      const armProgressTimer = () => {
        clearTimeout(progressTimer);
        progressTimer = setTimeout(
          () => stop(`no progress for ${this.noProgressMs / 1000} s`),
          this.noProgressMs,
        );
      };
      armProgressTimer();

      // IPC messages from worker
      child.on('message', (msg: WorkerOutMessage) => {
        switch (msg.type) {
          case 'ready':
            logger.info(P, `Worker ready: ${msg.modelId} (${msg.dimensions}d)`);
            armProgressTimer();
            break;
          case 'progress':
            armProgressTimer();
            embedded = msg.embedded;
            this.status.embedded = msg.embedded;
            this.status.total = msg.total;
            this.persistProgress(job.suffix, msg.embedded, msg.total);
            this.notifyStatusChange();
            break;
          case 'done':
            embedded = msg.embedded;
            logger.info(P, `Worker done: ${msg.embedded} chunks in ${msg.durationMs}ms`);
            // Resolve immediately on done — don't wait for exit.
            // ONNX runtime can crash on process.exit() (libc++abi mutex error),
            // which causes a non-zero exit code despite successful embedding.
            finish();
            break;
          case 'memory':
            this.status.rssMB = msg.rssMB;
            this.status.heapUsedMB = msg.heapUsedMB;
            break;
          case 'error':
            logger.error(P, `Worker error: ${msg.message}`);
            if (msg.fatal) {
              finish(msg.message);
            }
            break;
        }
      });

      // Process exit — only clear this.child if it still points to THIS child
      // (a new job may have already assigned a new child)
      child.on('exit', (code) => {
        if (this.child === child) this.child = null;
        if (stopping) {
          finish(stopping);
        } else if (code === 0) {
          finish();
        } else {
          finish(`Worker exited with code ${code}`);
        }
      });

      child.on('error', (err) => {
        if (this.child === child) this.child = null;
        finish(err.message);
      });

      child.on('disconnect', () => {
        // IPC channel broke — the child may still be running; stop it and wait for exit
        if (!resolved) stop('IPC channel disconnected');
      });

      // Send start message
      const startMsg: WorkerStartMessage = {
        type: 'start',
        dbPath: this.dbPath,
        suffix: job.suffix,
        embedderType: job.embedderType,
        config: job.config,
        batchSize,
        logLevel: job.logLevel,
        threads: EMBED_WORKER_THREADS,
      };
      child.send(startMsg);
    });
  }

  async runAllJobs(opts: {
    textEnabled: boolean;
    codeEnabled: boolean;
    logLevel?: string;
    config: {
      embeddingTextBackend: string;
      embeddingTextModel: string | null;
      embeddingTextWindows?: boolean;
      embeddingCodeBackend: string;
      embeddingCodeModel: string | null;
      ollamaBaseUrl: string;
    };
  }): Promise<boolean> {
    return this.runExclusive(() => this.runJobs(opts));
  }

  private async runJobs(opts: Parameters<EmbedOrchestrator['runAllJobs']>[0]): Promise<void> {
    if (opts.textEnabled) {
      const result = await this.runJob({
        suffix: '_text',
        embedderType: 'text',
        config: {
          embeddingBackend: opts.config.embeddingTextBackend as
            | 'auto'
            | 'transformers-js'
            | 'ollama',
          embeddingModel: opts.config.embeddingTextModel,
          ollamaBaseUrl: opts.config.ollamaBaseUrl,
          splitWindows: opts.config.embeddingTextWindows,
        },
        logLevel: opts.logLevel,
      });
      if (result.error) {
        logger.error(P, `Text embedding failed: ${result.error}`);
      } else if (result.embedded > 0) {
        logger.info(P, `Text embedding complete: ${result.embedded} chunks`);
      }
    }

    if (opts.codeEnabled) {
      const result = await this.runJob({
        suffix: '_code',
        embedderType: 'code',
        config: {
          embeddingBackend: opts.config.embeddingCodeBackend as
            | 'auto'
            | 'transformers-js'
            | 'ollama',
          embeddingModel: opts.config.embeddingCodeModel,
          ollamaBaseUrl: opts.config.ollamaBaseUrl,
        },
        batchSize: 10,
        logLevel: opts.logLevel,
      });
      if (result.error) {
        logger.error(P, `Code embedding failed: ${result.error}`);
      } else if (result.embedded > 0) {
        logger.info(P, `Code embedding complete: ${result.embedded} chunks`);
      }
    }
  }

  abort(): void {
    if (this.child) {
      const child = this.child;
      logger.info(P, `Aborting embed worker (pid=${child.pid})`);
      // SIGKILL directly — SIGTERM is unreliable when worker is blocked in native ONNX code.
      // The cooperative AbortSignal in embedChunks handles graceful shutdown between batches,
      // but during a batch the event loop is blocked and SIGTERM handlers can't execute.
      try {
        child.kill('SIGKILL');
      } catch {
        // Process already dead — ignore
      }
      this.child = null;
    }
    this.resetStatus();
  }

  private persistProgress(suffix: string, embedded: number, total: number): void {
    try {
      setMeta(this.db, `embed_progress_embedded${suffix}`, String(embedded));
      setMeta(this.db, `embed_progress_total${suffix}`, String(total));
    } catch {
      // DB write failure in progress tracking is non-fatal
    }
  }

  private cleanProgressMeta(suffix: string): void {
    try {
      deleteMeta(this.db, `embed_progress_embedded${suffix}`);
      deleteMeta(this.db, `embed_progress_total${suffix}`);
    } catch {
      // Non-fatal
    }
  }

  private cleanStaleProgress(): void {
    try {
      deleteMeta(this.db, 'embed_progress_embedded_text');
      deleteMeta(this.db, 'embed_progress_total_text');
      deleteMeta(this.db, 'embed_progress_embedded_code');
      deleteMeta(this.db, 'embed_progress_total_code');
    } catch {
      // Non-fatal
    }
  }

  private resetStatus(): void {
    this.status = {
      active: false,
      suffix: null,
      embedded: 0,
      total: 0,
      pid: null,
      rssMB: null,
      heapUsedMB: null,
    };
    this.notifyStatusChange();
  }

  private notifyStatusChange(): void {
    const status = this.getStatus();
    for (const cb of this.statusListeners) cb(status);
  }
}
