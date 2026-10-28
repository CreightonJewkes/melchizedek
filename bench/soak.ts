/**
 * Soak test on a database COPY: synthetic sessions grow and are re-indexed the way Stop hooks
 * do it (now and then a stale snapshot commits late), while the real orchestrator runs the BUILT
 * embed worker on a short continuous-embedding interval. Reports, over the run: the oldest
 * pending chunk's age against the freshness objective (one interval plus one job), orphan
 * vectors, dangling map rows, and stale vectors (stored vector vs. a fresh embedding of the
 * chunk's current text).
 *
 * Usage: tsx bench/soak.ts --db <copy.db> [--minutes 10] [--interval 60] [--sessions 6] [--turn 3]
 *
 * Needs `npm run build` (the worker runs from dist). Never point it at a live database.
 */

import { createRequire } from 'module';
import fs from 'fs';
import os from 'os';
import path from 'path';
import {
  openDatabase,
  hasPendingEmbeddings,
  oldestPendingEmbedding,
  maintainVectors,
  vectorHealth,
} from '../src/db.js';
import { indexConvSession } from '../src/indexer.js';
import { EmbedOrchestrator } from '../src/embed-orchestrator.js';
import { createTextEmbedder } from '../src/embedder.js';

// db.ts loads sqlite-vec with require(); provide one under tsx's ESM loader, as the harness does.
(globalThis as { require?: NodeRequire }).require = createRequire(import.meta.url);

const arg = (name: string, d: string): string => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : d;
};
const dbPath = arg('db', '');
if (!dbPath) {
  console.error('usage: tsx bench/soak.ts --db <copy.db>');
  process.exit(2);
}
const minutes = parseFloat(arg('minutes', '10'));
const intervalMs = parseFloat(arg('interval', '60')) * 1000;
const nSessions = parseInt(arg('sessions', '6'), 10);
const turnMs = parseFloat(arg('turn', '3')) * 1000;

const { db } = openDatabase(dbPath);
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'm9k-soak-'));
const worker = path.join(import.meta.dirname, '../dist/embed-worker.js');
const orch = new EmbedOrchestrator(dbPath, db, { workerPath: worker });
const jobs = {
  textEnabled: true,
  codeEnabled: false,
  logLevel: 'warn',
  config: {
    embeddingTextBackend: 'transformers-js',
    embeddingTextModel: null,
    embeddingCodeBackend: 'auto',
    embeddingCodeModel: null,
    ollamaBaseUrl: 'http://localhost:11434',
  },
};

// Distinct texts (random words), so a stale vector cannot pass for a fresh one.
const vocab = Array.from({ length: 400 }, (_, i) => `w${(i * 7919) % 10007}`);
let seed = 20261028;
const word = () => vocab[(seed = (seed * 48271) % 2147483647) % vocab.length];
const sentence = (n: number) => Array.from({ length: n }, word).join(' ');

const lines = new Map<string, string[]>();
const t0 = Date.now();
let turn = 0;
let staleCommits = 0;
let staleSkipped = 0;
const hookMs: number[] = [];

function oneTurn(): void {
  const id = `soak-${turn % nSessions}`;
  const file = path.join(dir, `${id}.jsonl`);
  const prev = lines.get(id) ?? [];
  const at = new Date().toISOString();
  const next = [
    ...prev,
    JSON.stringify({
      type: 'user',
      uuid: `u${turn}`,
      sessionId: id,
      cwd: '/soak',
      timestamp: at,
      message: { role: 'user', content: `turn ${turn} ${sentence(30)}` },
    }),
    JSON.stringify({
      type: 'assistant',
      uuid: `a${turn}`,
      parentUuid: `u${turn}`,
      sessionId: id,
      timestamp: at,
      message: { role: 'assistant', content: [{ type: 'text', text: sentence(40) }] },
    }),
  ];
  lines.set(id, next);
  fs.writeFileSync(file, next.join('\n'));
  const { mtimeMs } = fs.statSync(file);
  const h0 = performance.now();
  indexConvSession(db, id, fs.readFileSync(file, 'utf8'), '/soak', file, { mtimeMs });
  hookMs.push(performance.now() - h0);
  // Every tenth turn, a hook that read the previous snapshot commits late.
  if (turn % 10 === 9 && prev.length > 0) {
    staleCommits++;
    if (
      indexConvSession(db, id, prev.join('\n'), '/soak', file, { mtimeMs: mtimeMs - 1 }) ===
      'skipped'
    ) {
      staleSkipped++;
    }
  }
  turn++;
}

const samples: Array<{ s: number; pendingAgeS: number | null; busy: boolean }> = [];
const jobSecs: number[] = [];
async function embedIfNeeded(): Promise<void> {
  if (orch.isBusy() || !hasPendingEmbeddings(db, '_text')) return;
  const j0 = Date.now();
  if (await orch.runAllJobs(jobs)) jobSecs.push((Date.now() - j0) / 1000);
}

await orch.runExclusive(async () => void maintainVectors(db));
const before = vectorHealth(db).find((v) => v.table === 'conv_vec_text');
await embedIfNeeded(); // the copy's own backlog first, so the soak measures fresh chunks only
const turnTimer = setInterval(oneTurn, turnMs);
const embedTimer = setInterval(() => void embedIfNeeded(), intervalMs);
const sampleTimer = setInterval(() => {
  const since = oldestPendingEmbedding(db, '_text');
  samples.push({
    s: Math.round((Date.now() - t0) / 1000),
    pendingAgeS: since ? Math.round((Date.now() - Date.parse(since)) / 1000) : null,
    busy: orch.isBusy(),
  });
}, 5_000);
await new Promise((r) => setTimeout(r, minutes * 60_000));
clearInterval(turnTimer);
clearInterval(embedTimer);
clearInterval(sampleTimer);
while (orch.isBusy()) await new Promise((r) => setTimeout(r, 1_000));
await embedIfNeeded(); // drain the last turns, then check
while (orch.isBusy()) await new Promise((r) => setTimeout(r, 1_000));

// Stale vectors: every mapped soak chunk's stored vector against a fresh embedding.
const embedder = (await createTextEmbedder({
  embeddingBackend: 'transformers-js',
  embeddingModel: null,
}))!;
const mapped = db
  .prepare(
    `SELECT c.user_content || ' ' || c.assistant_content AS content, v.embedding
     FROM conv_chunks c JOIN conv_vec_map_text m ON m.chunk_id = c.id
     JOIN conv_vec_text v ON v.rowid = m.vec_rowid WHERE c.session_id LIKE 'soak-%'`,
  )
  .all() as Array<{ content: string; embedding: Buffer }>;
let stale = 0;
let minCos = 1;
for (const r of mapped) {
  const stored = new Float32Array(
    r.embedding.buffer,
    r.embedding.byteOffset,
    r.embedding.length / 4,
  );
  const fresh = await embedder.embed(r.content);
  const cos = stored.reduce((a, x, i) => a + x * fresh[i], 0);
  minCos = Math.min(minCos, cos);
  if (cos < 0.97) stale++;
}
const after = vectorHealth(db).find((v) => v.table === 'conv_vec_text');
const ages = samples.map((x) => x.pendingAgeS ?? 0);
const maxJob = Math.max(0, ...jobSecs);
const hs = [...hookMs].sort((a, b) => a - b);
const objectiveS = intervalMs / 1000 + maxJob + 5; // one interval + one job + the sampling period
console.log(
  JSON.stringify(
    {
      minutes,
      intervalS: intervalMs / 1000,
      turns: turn,
      soakChunks: (
        db
          .prepare("SELECT COUNT(*) AS n FROM conv_chunks WHERE session_id LIKE 'soak-%'")
          .get() as { n: number }
      ).n,
      staleCommits,
      staleSkipped,
      jobs: jobSecs.length,
      jobSecsMax: +maxJob.toFixed(1),
      pendingAgeMaxS: Math.max(0, ...ages),
      freshnessObjectiveS: +objectiveS.toFixed(1),
      freshnessMet: Math.max(0, ...ages) <= objectiveS,
      orphanVectors: { before: before?.orphanVectors, after: after?.orphanVectors },
      danglingMaps: after?.danglingMaps,
      mappedSoakChunks: mapped.length,
      staleVectors: stale,
      minCosine: +minCos.toFixed(4),
      hookMs: {
        p50: +hs[Math.floor(hs.length / 2)].toFixed(2),
        p95: +hs[Math.floor(hs.length * 0.95)].toFixed(2),
      },
      lastJobError: orch.getLastError(),
    },
    null,
    1,
  ),
);
db.close();
fs.rmSync(dir, { recursive: true, force: true });
