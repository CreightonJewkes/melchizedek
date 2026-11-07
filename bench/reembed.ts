/**
 * Re-embed a database COPY with another text model through the real migration path — the
 * same calls the embed worker makes (seed meta, resume or detect the model change, embed into
 * the next generation, transactional swap) — so it doubles as a rehearsal of a live switch.
 *
 * Usage: tsx bench/reembed.ts --db <copy.db> --model <registry key> [--batch 50] [--windows]
 *
 * --windows re-embeds with split prompt/reply windows (src/windows.ts), a recipe change.
 *
 * Model files are read from M9K_MODEL_CACHE when set; remote downloads stay off. Set
 * M9K_ONNX_THREADS to cap inference threads, as the daemon's worker does. Never point it at a
 * live database. A killed run resumes from its checkpoint when started again.
 */

import { createRequire } from 'module';
import { env } from '@huggingface/transformers';
import {
  openDatabase,
  createVecTables,
  countEligibleChunks,
  countEmbeddedChunks,
} from '../src/db.js';
import { createTextEmbedder } from '../src/embedder.js';
import { seedEmbeddingMeta, detectModelChange, EmbeddingMigration } from '../src/migration.js';

// db.ts loads sqlite-vec with require(); provide one under tsx's ESM loader, as the harness does.
(globalThis as { require?: NodeRequire }).require = createRequire(import.meta.url);
if (process.env.M9K_MODEL_CACHE) env.cacheDir = process.env.M9K_MODEL_CACHE;
env.allowRemoteModels = false;

const arg = (name: string, d?: string): string | undefined => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : d;
};
const dbPath = arg('db');
const model = arg('model');
if (!dbPath || !model) {
  console.error('usage: tsx bench/reembed.ts --db <copy.db> --model <registry key>');
  process.exit(2);
}
const batchSize = parseInt(arg('batch', '50')!, 10);

const { db, vecEnabled } = openDatabase(dbPath);
if (!vecEnabled) throw new Error('sqlite-vec not available');
const embedder = await createTextEmbedder({
  embeddingBackend: 'transformers-js',
  embeddingModel: model,
  windows: process.argv.includes('--windows'),
});
if (!embedder) throw new Error(`model ${model} not available`);

const t0 = performance.now();
let last = 0;
const onProgress = (embedded: number, total: number) => {
  if (performance.now() - last < 60_000) return;
  last = performance.now();
  const s = (last - t0) / 1000;
  console.log(
    JSON.stringify({ embedded, total, secs: Math.round(s), per_s: +(embedded / s).toFixed(1) }),
  );
};

createVecTables(db, embedder.dimensions(), '_text');
seedEmbeddingMeta(db, embedder, '_text');
const migration =
  EmbeddingMigration.resume(db, embedder, { suffix: '_text', batchSize }) ??
  (detectModelChange(db, embedder, '_text')
    ? new EmbeddingMigration(db, embedder, { suffix: '_text', batchSize, onProgress })
    : null);
if (!migration) {
  console.log(JSON.stringify({ note: `already on ${embedder.modelId()}; nothing to do` }));
  process.exit(0);
}
migration.setOnProgress(onProgress);
const result = await migration.run();
const secs = (performance.now() - t0) / 1000;
console.log(
  JSON.stringify({
    model: embedder.modelId(),
    dims: embedder.dimensions(),
    threads: process.env.M9K_ONNX_THREADS ?? 'default',
    ...result,
    secs: Math.round(secs),
    per_s: +(result.embedded / secs).toFixed(1),
    eligible: countEligibleChunks(db),
    mapped: countEmbeddedChunks(db, '_text'),
  }),
);
db.close();
