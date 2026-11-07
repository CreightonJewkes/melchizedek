/**
 * Embedding model migration: dual-index zero-downtime re-embedding.
 *
 * State machine: prepare → reembed → swap | abort
 * Meta keys: embedding_model_id, embedding_dimensions, migration_*
 */

import type Database from 'better-sqlite3';
import type { Embedder } from './models.js';
import {
  getMeta,
  setMeta,
  deleteMeta,
  createVecTables,
  recreateVecTables,
  getVecTableDimensions,
  countEmbeddedChunks,
  publishVectors,
  MIN_EMBED_CONTENT_LENGTH,
} from './db.js';
import { logger } from './logger.js';
import { embedChunkWindows, secondWindowSuffix } from './windows.js';

const P = 'migration';

// --- Seed + Detect ---

export function embedderRecipe(embedder: Embedder): string {
  return embedder.recipe?.() ?? '';
}

export function seedEmbeddingMeta(
  db: Database.Database,
  embedder: Embedder | null,
  suffix = '_text',
): void {
  if (!embedder) return;
  const existing = getMeta(db, `embedding_model_id${suffix}`);
  if (!existing) {
    setMeta(db, `embedding_model_id${suffix}`, embedder.modelId());
    setMeta(db, `embedding_dimensions${suffix}`, String(embedder.dimensions()));
    setMeta(db, `embedding_recipe${suffix}`, embedderRecipe(embedder));
  } else if (existing === embedder.modelId() && !getMeta(db, `embedding_recipe${suffix}`)) {
    // Databases from before recipe tags were built with the registry recipe of the day.
    setMeta(db, `embedding_recipe${suffix}`, embedderRecipe(embedder));
  }
}

export interface ModelChange {
  oldModelId: string;
  newModelId: string;
  oldDimensions: number;
  newDimensions: number;
}

/** A stored recipe that is missing reads as the embedder's current registry recipe. */
function sameRecipe(db: Database.Database, embedder: Embedder, key: string): boolean {
  const stored = getMeta(db, key);
  return stored === null || stored === embedderRecipe(embedder);
}

export function detectModelChange(
  db: Database.Database,
  embedder: Embedder,
  suffix = '_text',
): ModelChange | null {
  const storedModelId = getMeta(db, `embedding_model_id${suffix}`);
  if (!storedModelId) return null; // First time — no change to detect

  if (
    storedModelId === embedder.modelId() &&
    sameRecipe(db, embedder, `embedding_recipe${suffix}`)
  ) {
    return null; // Same model, same recipe
  }

  const storedDimensions = parseInt(getMeta(db, `embedding_dimensions${suffix}`) ?? '0', 10);

  return {
    oldModelId: storedModelId,
    newModelId: embedder.modelId(),
    oldDimensions: storedDimensions,
    newDimensions: embedder.dimensions(),
  };
}

/**
 * True when this generation's vectors were made by this embedder's exact recipe and
 * no migration is replacing them. Search uses it to skip the vector channel, keeping
 * keyword search, rather than compare query and index vectors from different models.
 */
export function vectorIndexMatches(
  db: Database.Database,
  embedder: Embedder,
  suffix = '_text',
): boolean {
  if (getMeta(db, `migration_target_model${suffix}`)) return false;
  const storedModelId = getMeta(db, `embedding_model_id${suffix}`);
  if (storedModelId === null) return true; // nothing embedded under any model yet
  return (
    storedModelId === embedder.modelId() && sameRecipe(db, embedder, `embedding_recipe${suffix}`)
  );
}

/**
 * Make the primary vec tables fit this embedder without destroying vectors. Missing or
 * empty tables of the wrong size are (re)created. Populated ones are kept: the embed
 * worker re-embeds into the next generation and swaps atomically, and search skips
 * vectors until then (vectorIndexMatches).
 */
export function reconcileVecTables(
  db: Database.Database,
  embedder: Embedder,
  suffix = '_text',
): void {
  const tableDims = getVecTableDimensions(db, suffix);
  const expected = embedder.dimensions();
  if (tableDims === expected) return;
  if (tableDims === null) {
    createVecTables(db, expected, suffix);
    return;
  }
  const mapped = countEmbeddedChunks(db, suffix);
  if (mapped === 0) {
    logger.info(P, `Recreating empty vec tables${suffix} for ${expected}d embeddings`);
    recreateVecTables(db, expected, suffix);
    return;
  }
  if (!getMeta(db, `embedding_model_id${suffix}`)) {
    // Vectors of unknown origin: record them so the worker migrates instead of mixing.
    setMeta(db, `embedding_model_id${suffix}`, 'unknown-legacy');
    setMeta(db, `embedding_dimensions${suffix}`, String(tableDims));
  }
  logger.warn(
    P,
    `Vec tables${suffix} hold ${mapped} vectors at ${tableDims}d; keeping them until the ` +
      `embed worker migrates to ${expected}d (vector search paused meanwhile)`,
  );
}

// --- Migration result ---

export interface MigrationResult {
  embedded: number;
  aborted: boolean;
  swapped: boolean;
}

// --- Migration lock ---

export function acquireMigrationLock(
  db: Database.Database,
  instanceId?: string,
  suffix = '_text',
): boolean {
  // Check and claim in one write transaction, so two workers cannot both win.
  return db.transaction(() => claimMigrationLock(db, instanceId, suffix)).immediate();
}

function claimMigrationLock(
  db: Database.Database,
  instanceId: string | undefined,
  suffix: string,
): boolean {
  const existingLock = getMeta(db, `migration_lock${suffix}`);
  if (existingLock) {
    // Check if the lock holder is a dead process (stale lock)
    const pidMatch = existingLock.match(/^pid-(\d+)$/);
    if (pidMatch) {
      const pid = parseInt(pidMatch[1], 10);
      try {
        process.kill(pid, 0); // Throws if process doesn't exist
        return false; // Process alive — lock is valid
      } catch {
        // Process dead — stale lock, clean it up
        logger.warn(P, `Cleaning stale lock from dead process ${pid}`);
        deleteMeta(db, `migration_lock${suffix}`);
      }
    } else {
      return false; // Non-PID lock, respect it
    }
  }
  setMeta(db, `migration_lock${suffix}`, instanceId ?? `pid-${process.pid}`);
  return true;
}

export function releaseMigrationLock(db: Database.Database, suffix = '_text'): void {
  deleteMeta(db, `migration_lock${suffix}`);
}

// --- Migration engine ---

export class EmbeddingMigration {
  private db: Database.Database;
  private embedder: Embedder;
  private batchSize: number;
  private signal?: AbortSignal;
  private onProgress?: (embedded: number, total: number) => void;
  private suffix: string;

  constructor(
    db: Database.Database,
    embedder: Embedder,
    options?: {
      batchSize?: number;
      signal?: AbortSignal;
      onProgress?: (embedded: number, total: number) => void;
      suffix?: string;
    },
  ) {
    this.db = db;
    this.embedder = embedder;
    this.batchSize = options?.batchSize ?? 50;
    this.signal = options?.signal;
    this.onProgress = options?.onProgress;
    this.suffix = options?.suffix ?? '_text';
  }

  /** Set onProgress callback (useful when resuming a migration) */
  setOnProgress(cb: (embedded: number, total: number) => void): void {
    this.onProgress = cb;
  }

  /** Phase 1: Create _next tables + set migration meta */
  prepare(): void {
    const targetModel = this.embedder.modelId();
    const targetDimensions = this.embedder.dimensions();

    createVecTables(this.db, targetDimensions, `${this.suffix}_next`);
    if (this.embedder.splitWindows?.()) {
      createVecTables(this.db, targetDimensions, secondWindowSuffix(`${this.suffix}_next`));
    }

    setMeta(this.db, `migration_target_model${this.suffix}`, targetModel);
    setMeta(this.db, `migration_target_dimensions${this.suffix}`, String(targetDimensions));
    setMeta(this.db, `migration_target_recipe${this.suffix}`, embedderRecipe(this.embedder));
    setMeta(this.db, `migration_progress${this.suffix}`, '0');
    setMeta(this.db, `migration_started_at${this.suffix}`, new Date().toISOString());
  }

  /** Phase 2: Re-embed all eligible chunks into _next tables */
  async reembed(): Promise<{ embedded: number; aborted: boolean }> {
    let totalEmbedded = parseInt(getMeta(this.db, `migration_progress${this.suffix}`) ?? '0', 10);

    const totalEligible = (
      this.db
        .prepare(
          `SELECT COUNT(*) AS cnt FROM conv_chunks
           WHERE deleted_at IS NULL
             AND LENGTH(user_content) + LENGTH(assistant_content) >= ?`,
        )
        .get(MIN_EMBED_CONTENT_LENGTH) as { cnt: number }
    ).cnt;

    logger.debug(
      P,
      `reembed${this.suffix}: ${totalEmbedded} already done, ${totalEligible} eligible`,
    );

    let batch = this.getNextBatch();
    let previousIds = '';
    logger.debug(P, `reembed${this.suffix}: first batch size=${batch.length}`);
    while (batch.length > 0) {
      if (this.signal?.aborted) {
        return { embedded: totalEmbedded, aborted: true };
      }

      const ids = batch.map((c) => `${c.id}:${c.hash}`).join(',');
      if (ids === previousIds) throw new Error('re-embed made no progress on the same batch');

      logger.debug(P, `reembed${this.suffix}: embedding batch of ${batch.length} chunks`);
      const count = await this.reembedBatch(batch);
      totalEmbedded += count;
      previousIds = count === 0 ? ids : '';
      logger.debug(P, `reembed${this.suffix}: batch done, total=${totalEmbedded}/${totalEligible}`);

      // Checkpoint
      setMeta(this.db, `migration_progress${this.suffix}`, String(totalEmbedded));
      this.onProgress?.(totalEmbedded, totalEligible);

      batch = this.getNextBatch();
    }

    logger.debug(P, `reembed${this.suffix}: complete, total=${totalEmbedded}`);
    return { embedded: totalEmbedded, aborted: false };
  }

  /** Embed a single batch into _next tables */
  async reembedBatch(
    batch?: Array<{ id: string; content: string; hash: string; user: string; assistant: string }>,
  ): Promise<number> {
    const items = batch ?? this.getNextBatch();
    if (items.length === 0) return 0;

    const texts = items.map((c) => c.content);
    const lengths = texts.map((t) => t.length);
    const minLen = Math.min(...lengths);
    const maxLen = Math.max(...lengths);
    const avgLen = Math.round(lengths.reduce((a, b) => a + b, 0) / lengths.length);
    logger.debug(
      P,
      `reembedBatch${this.suffix}: ${texts.length} texts, len min=${minLen} max=${maxLen} avg=${avgLen}`,
    );
    const embeddings = await embedChunkWindows(this.embedder, items);

    const vectorItems = items.map((c, i) => ({ chunkId: c.id, hash: c.hash, ...embeddings[i] }));

    // Stale items (changed or deleted during inference) are discarded and re-read.
    return publishVectors(this.db, vectorItems, `${this.suffix}_next`);
  }

  /**
   * Phase 3: publish the next generation atomically. Old tables, the copy, the _next
   * drop and the model meta change in ONE transaction (sqlite-vec >= 0.1.9 supports
   * vec0 DDL inside it), so a failure at any step leaves the old generation searchable
   * and the migration resumable. Only mapped vectors are copied.
   */
  swap(): void {
    const s = this.suffix;
    const targetModel = getMeta(this.db, `migration_target_model${s}`)!;
    const targetDimensions = parseInt(getMeta(this.db, `migration_target_dimensions${s}`)!, 10);
    const targetRecipe =
      getMeta(this.db, `migration_target_recipe${s}`) ?? embedderRecipe(this.embedder);

    const w2 = secondWindowSuffix(s);
    const w2Next = secondWindowSuffix(`${s}_next`);
    const nextHasW2 = !!this.db
      .prepare("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?")
      .get(`conv_vec_map${w2Next}`);

    this.db
      .transaction(() => {
        // The second-window pair belongs to the old recipe: replaced, or dropped when the
        // new one has no windows.
        this.db.exec(`
          DROP TABLE IF EXISTS conv_vec_map${s};
          DROP TABLE IF EXISTS conv_vec${s};
          DROP TABLE IF EXISTS conv_vec_map${w2};
          DROP TABLE IF EXISTS conv_vec${w2};
        `);

        // CRITICAL: sqlite-vec virtual tables cannot be safely renamed with
        // ALTER TABLE RENAME — the backing tables (_info, _chunks, _rowids,
        // _vector_chunks00) keep their old names, breaking INSERT/SELECT.
        // Instead: create fresh vec table, copy data, then drop _next.
        createVecTables(this.db, targetDimensions, s);

        this.db.exec(`
          INSERT INTO conv_vec${s}(rowid, embedding)
            SELECT v.rowid, v.embedding FROM conv_vec_map${s}_next m
            JOIN conv_vec${s}_next v ON v.rowid = m.vec_rowid;

          INSERT INTO conv_vec_map${s}(vec_rowid, chunk_id)
            SELECT vec_rowid, chunk_id FROM conv_vec_map${s}_next;
        `);

        if (nextHasW2) {
          createVecTables(this.db, targetDimensions, w2);
          this.db.exec(`
            INSERT INTO conv_vec${w2}(rowid, embedding)
              SELECT v.rowid, v.embedding FROM conv_vec_map${w2Next} m
              JOIN conv_vec${w2Next} v ON v.rowid = m.vec_rowid;
            INSERT INTO conv_vec_map${w2}(vec_rowid, chunk_id)
              SELECT vec_rowid, chunk_id FROM conv_vec_map${w2Next};
          `);
        }

        this.db.exec(`
          DROP TABLE IF EXISTS conv_vec_map${s}_next;
          DROP TABLE IF EXISTS conv_vec${s}_next;
          DROP TABLE IF EXISTS conv_vec_map${w2Next};
          DROP TABLE IF EXISTS conv_vec${w2Next};
        `);

        setMeta(this.db, `embedding_model_id${s}`, targetModel);
        setMeta(this.db, `embedding_dimensions${s}`, String(targetDimensions));
        setMeta(this.db, `embedding_recipe${s}`, targetRecipe);
        // The rebuilt table has no dead slots.
        deleteMeta(this.db, `vec_dead_rows:conv_vec${s}`);
        deleteMeta(this.db, `vec_dead_rows:conv_vec${w2}`);
        this.cleanupMigrationMeta();
      })
      .immediate();
  }

  /** Abort: drop _next tables, clean up meta */
  abort(): void {
    try {
      const w2Next = secondWindowSuffix(`${this.suffix}_next`);
      this.db.exec(`
        DROP TABLE IF EXISTS conv_vec_map${this.suffix}_next;
        DROP TABLE IF EXISTS conv_vec${this.suffix}_next;
        DROP TABLE IF EXISTS conv_vec_map${w2Next};
        DROP TABLE IF EXISTS conv_vec${w2Next};
      `);
    } catch (err) {
      // sqlite-vec virtual tables can fail to DROP — don't let that block meta cleanup
      logger.warn(P, `Failed to drop _next tables for ${this.suffix} (non-fatal):`, err);
    }
    this.cleanupMigrationMeta();
  }

  private cleanupMigrationMeta(): void {
    deleteMeta(this.db, `migration_target_model${this.suffix}`);
    deleteMeta(this.db, `migration_target_dimensions${this.suffix}`);
    deleteMeta(this.db, `migration_target_recipe${this.suffix}`);
    deleteMeta(this.db, `migration_progress${this.suffix}`);
    deleteMeta(this.db, `migration_started_at${this.suffix}`);
    deleteMeta(this.db, `vec_dead_rows:conv_vec${this.suffix}_next`);
    deleteMeta(this.db, `vec_dead_rows:conv_vec${secondWindowSuffix(`${this.suffix}_next`)}`);
  }

  /** Full lifecycle: prepare → reembed → swap | abort */
  async run(): Promise<MigrationResult> {
    logger.debug(P, `Migration.run() starting for ${this.suffix}`);
    const locked = acquireMigrationLock(this.db, undefined, this.suffix);
    if (!locked) {
      logger.warn(P, `Migration.run() failed to acquire lock for ${this.suffix}`);
      return { embedded: 0, aborted: true, swapped: false };
    }
    logger.debug(P, `Lock acquired for ${this.suffix}`);

    try {
      // Check if prepare is needed (resuming skips prepare)
      if (!getMeta(this.db, `migration_target_model${this.suffix}`)) {
        logger.debug(P, `Preparing migration for ${this.suffix}`);
        this.prepare();
      } else {
        // Verify _next tables still exist — they may have been dropped by a SIGKILL
        const nextTableExists = this.db
          .prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name=?`)
          .get(`conv_vec_map${this.suffix}_next`);
        if (!nextTableExists) {
          logger.debug(P, `_next tables missing, re-preparing for ${this.suffix}`);
          this.prepare();
        } else {
          logger.debug(P, `Resuming existing migration for ${this.suffix}`);
        }
      }

      logger.debug(P, `Starting reembed for ${this.suffix}`);
      const { embedded, aborted } = await this.reembed();

      if (aborted) {
        // Cancelled (shutdown or timeout): keep _next and the checkpoint, so the next
        // run resumes instead of re-embedding everything.
        return { embedded, aborted: true, swapped: false };
      }

      this.swap();
      return { embedded, aborted: false, swapped: true };
    } catch (err) {
      // Clean up on ANY error — don't leave migration meta stuck forever
      logger.error(P, `Error during ${this.suffix} migration, aborting:`, err);
      try {
        this.abort();
      } catch {
        /* ignore cleanup errors */
      }
      return { embedded: 0, aborted: true, swapped: false };
    } finally {
      releaseMigrationLock(this.db, this.suffix);
    }
  }

  /** Resume an in-progress migration or start fresh if target model changed */
  static resume(
    db: Database.Database,
    embedder: Embedder,
    options?: {
      batchSize?: number;
      signal?: AbortSignal;
      onProgress?: (embedded: number, total: number) => void;
      suffix?: string;
    },
  ): EmbeddingMigration | null {
    const suffix = options?.suffix ?? '_text';
    const targetModel = getMeta(db, `migration_target_model${suffix}`);
    if (!targetModel) return null; // No migration in progress

    const migration = new EmbeddingMigration(db, embedder, options);

    const targetRecipe = getMeta(db, `migration_target_recipe${suffix}`);
    if (
      targetModel !== embedder.modelId() ||
      (targetRecipe !== null && targetRecipe !== embedderRecipe(embedder))
    ) {
      // Target changed — abort old migration, return fresh instance
      migration.abort();
      return new EmbeddingMigration(db, embedder, options);
    }

    return migration;
  }

  /** Get next batch of chunks not yet in _next tables */
  private getNextBatch(): Array<{
    id: string;
    content: string;
    hash: string;
    user: string;
    assistant: string;
  }> {
    return this.db
      .prepare(
        `SELECT c.id, (c.user_content || ' ' || c.assistant_content) AS content, c.hash,
                c.user_content AS user, c.assistant_content AS assistant
         FROM conv_chunks c
         LEFT JOIN conv_vec_map${this.suffix}_next m ON c.id = m.chunk_id
         WHERE m.chunk_id IS NULL
           AND c.deleted_at IS NULL
           AND LENGTH(c.user_content) + LENGTH(c.assistant_content) >= ?
         ORDER BY c.timestamp DESC
         LIMIT ?`,
      )
      .all(MIN_EMBED_CONTENT_LENGTH, this.batchSize) as Array<{
      id: string;
      content: string;
      hash: string;
      user: string;
      assistant: string;
    }>;
  }
}
