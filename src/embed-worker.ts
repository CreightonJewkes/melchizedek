/**
 * Child process embed worker — loads one model, embeds all pending chunks, then exits.
 * Spawned by EmbedOrchestrator via child_process.fork().
 * Communicates with parent via IPC messages (WorkerInMessage / WorkerOutMessage).
 */

import os from 'os';
import { openDatabase, createVecTables } from './db.js';
import { createTextEmbedder, createCodeEmbedder } from './embedder.js';
import { seedEmbeddingMeta, detectModelChange, EmbeddingMigration } from './migration.js';
import { embedChunks } from './indexer.js';
import { initLogger, logger } from './logger.js';
import type { WorkerInMessage, WorkerOutMessage } from './models.js';

const P = 'embed-worker';

/** Worker debug lines go through the logger, so the configured level applies to them. */
function dbg(msg: string): void {
  logger.debug(P, msg);
}

function logMemory(label: string): void {
  const mem = process.memoryUsage();
  const rss = (mem.rss / 1024 / 1024).toFixed(1);
  const heap = (mem.heapUsed / 1024 / 1024).toFixed(1);
  logger.info(P, `RSS=${rss} MB, heap=${heap} MB — ${label}`);
}

function migrationFailureMessage(
  suffix: string,
  result: { embedded: number; aborted: boolean },
): string {
  return shutdownController.signal.aborted
    ? `migration${suffix} paused at ${result.embedded} chunks; resumes on next run`
    : `migration${suffix} failed after ${result.embedded} chunks; old index kept`;
}

// After SIGTERM the channel is closed: a send or a second disconnect would throw and turn
// a clean stop into the fatal-error exit (and ONNX's abort on process.exit).
function send(msg: WorkerOutMessage): void {
  if (process.send && process.connected) {
    process.send(msg);
  }
}

// Abort controller for cooperative shutdown between batches
const shutdownController = new AbortController();

// Periodic memory reporting interval (cleared on shutdown/done)
let memoryInterval: ReturnType<typeof setInterval> | null = null;

function sendMemoryReport(): void {
  const mem = process.memoryUsage();
  send({
    type: 'memory',
    rssMB: Math.round((mem.rss / 1024 / 1024) * 10) / 10,
    heapUsedMB: Math.round((mem.heapUsed / 1024 / 1024) * 10) / 10,
  });
}

function clearMemoryInterval(): void {
  if (memoryInterval) {
    clearInterval(memoryInterval);
    memoryInterval = null;
  }
}

// Handle shutdown signals — set abort flag, disconnect IPC, force exit after delay.
// process.exit() may trigger ONNX Runtime cleanup warnings, which is harmless during shutdown.
process.on('SIGTERM', () => {
  logger.info(P, 'SIGTERM received, shutting down...');
  clearMemoryInterval();
  shutdownController.abort();
  if (process.disconnect) process.disconnect();
  // Force exit after delay in case the event loop is blocked in native code
  setTimeout(() => process.exit(0), 2000).unref();
});

process.on('message', async (msg: WorkerInMessage) => {
  if (msg.type !== 'start') return;

  const startTime = Date.now();

  // Background work: the lowest CPU priority and capped inference threads, set by the
  // worker on itself so no other process is ever reniced.
  try {
    os.setPriority(0, os.constants.priority.PRIORITY_LOW); // 19, the lowest
  } catch {
    // not permitted here: keep the inherited priority
  }
  if (msg.threads && !process.env.M9K_ONNX_THREADS) {
    process.env.M9K_ONNX_THREADS = String(msg.threads);
  }

  try {
    // Initialize logger with parent's log level
    // stderr only (inherited from the daemon): the daemon owns the rotating log file.
    initLogger({
      level: (msg.logLevel ?? 'warn') as 'debug' | 'info' | 'warn' | 'error',
      file: false,
    });

    logger.debug(
      P,
      `Starting ${msg.embedderType} worker (suffix=${msg.suffix}, batchSize=${msg.batchSize})`,
    );

    // Open DB in WAL mode (concurrent reads from main, writes from here)
    const { db, vecEnabled } = openDatabase(msg.dbPath);
    logger.debug(P, `DB opened (vecEnabled=${vecEnabled})`);

    if (!vecEnabled) {
      send({ type: 'error', message: 'sqlite-vec not available', fatal: true });
      process.exit(1);
    }

    // Create the appropriate embedder
    logger.debug(
      P,
      `Creating ${msg.embedderType} embedder (backend=${msg.config.embeddingBackend}, model=${msg.config.embeddingModel})`,
    );
    const embedderConfig = {
      embeddingBackend: msg.config.embeddingBackend,
      embeddingModel: msg.config.embeddingModel,
      ollamaBaseUrl: msg.config.ollamaBaseUrl,
    };
    const embedder =
      msg.embedderType === 'text'
        ? await createTextEmbedder({ ...embedderConfig, windows: msg.config.splitWindows })
        : await createCodeEmbedder(embedderConfig);

    if (!embedder) {
      send({ type: 'error', message: `${msg.embedderType} embedder not available`, fatal: true });
      process.exit(1);
    }

    logMemory(`after ${msg.embedderType} model load`);
    dbg(`Embedder ready: ${embedder.modelId()} (${embedder.dimensions()}d)`);

    send({
      type: 'ready',
      modelId: embedder.modelId(),
      dimensions: embedder.dimensions(),
    });

    // Start periodic memory reporting (every 5s)
    sendMemoryReport();
    memoryInterval = setInterval(sendMemoryReport, 5000);

    // Ensure vec tables exist for this suffix (code tables aren't created by initSchema)
    dbg(`Creating vec tables for ${msg.suffix} (${embedder.dimensions()}d)`);
    createVecTables(db, embedder.dimensions(), msg.suffix);

    // Seed meta (idempotent)
    dbg(`seedEmbeddingMeta for ${msg.suffix}`);
    seedEmbeddingMeta(db, embedder, msg.suffix);

    // Check for model migration
    dbg(`Checking for migration resume (suffix=${msg.suffix})`);
    const resumed = EmbeddingMigration.resume(db, embedder, {
      suffix: msg.suffix,
      batchSize: msg.batchSize,
      signal: shutdownController.signal,
    });
    const modelChange = detectModelChange(db, embedder, msg.suffix);
    dbg(`Migration check: resumed=${!!resumed}, modelChange=${!!modelChange}`);

    let totalEmbedded = 0;
    let migrationFailure: string | null = null;

    const migrationProgress = (embedded: number, total: number) => {
      send({ type: 'progress', embedded, total });
    };

    if (resumed) {
      dbg(`Resuming ${msg.suffix} migration...`);
      resumed.setOnProgress(migrationProgress);
      dbg(`Calling migration.run() for ${msg.suffix}`);
      const result = await resumed.run();
      totalEmbedded = result.embedded;
      dbg(`Migration ${result.swapped ? 'completed' : 'aborted'}: ${result.embedded} chunks`);
      if (!result.swapped) migrationFailure = migrationFailureMessage(msg.suffix, result);
    } else if (modelChange) {
      dbg(`Model changed: ${modelChange.oldModelId} → ${modelChange.newModelId}`);
      const migration = new EmbeddingMigration(db, embedder, {
        suffix: msg.suffix,
        batchSize: msg.batchSize,
        onProgress: migrationProgress,
        signal: shutdownController.signal,
      });
      dbg(`Calling migration.run() for ${msg.suffix} (fresh)`);
      const result = await migration.run();
      totalEmbedded = result.embedded;
      dbg(`Migration ${result.swapped ? 'completed' : 'aborted'}: ${result.embedded} chunks`);
      if (!result.swapped) migrationFailure = migrationFailureMessage(msg.suffix, result);
    } else {
      // Normal backfill embedding
      dbg(`Starting backfill embedding for ${msg.suffix}`);
      totalEmbedded = await embedChunks(
        db,
        embedder,
        msg.suffix,
        msg.batchSize,
        (embedded, total) => {
          send({ type: 'progress', embedded, total });
        },
        shutdownController.signal,
      );
    }

    logMemory(`after ${msg.embedderType} embedding complete`);

    clearMemoryInterval();
    const durationMs = Date.now() - startTime;
    // A migration that did not publish is not "done": report it so status stays honest.
    if (migrationFailure) send({ type: 'error', message: migrationFailure, fatal: true });
    else send({ type: 'done', embedded: totalEmbedded, durationMs });

    // Close DB cleanly before exit
    db.close();

    // Disconnect IPC channel and let the process exit naturally.
    // Avoid process.exit(0) — ONNX runtime cleanup crashes with
    // "libc++abi: mutex lock failed" when exit is called explicitly.
    if (process.connected) process.disconnect();
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    logger.error(P, 'Fatal error:', err);
    send({ type: 'error', message, fatal: true });
    process.exit(1);
  }
});
