/**
 * Sweep and compact the vector tables of a database COPY (B2b), timing vector search before
 * and after: unfiltered KNN and the exact filtered scan, with seeded random unit queries.
 *
 * Usage: tsx bench/vector-maintenance.ts --db <copy.db> [--queries 30] [--since 2026-09-01]
 *
 * Never point it at a live database: it deletes orphan vectors and rewrites vec0 tables.
 */

import { createRequire } from 'module';
import {
  openDatabase,
  getVecTableDimensions,
  maintainVectors,
  deadVectorShare,
} from '../src/db.js';
import { searchVectorized } from '../src/search.js';

// db.ts loads sqlite-vec with require(); provide one under tsx's ESM loader, as the harness does.
(globalThis as { require?: NodeRequire }).require = createRequire(import.meta.url);

const arg = (name: string, d?: string): string | undefined => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : d;
};
const dbPath = arg('db');
if (!dbPath) {
  console.error('usage: tsx bench/vector-maintenance.ts --db <copy.db>');
  process.exit(2);
}
const n = parseInt(arg('queries', '30')!, 10);
const since = new Date(arg('since', '2026-09-01')!).toISOString();

const { db } = openDatabase(dbPath);
const dims = getVecTableDimensions(db, '_text') ?? 384;

let seed = 20261028;
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31) * 2 - 1;
const queries = Array.from({ length: n }, () => {
  const v = Float32Array.from({ length: dims }, rand);
  const norm = Math.hypot(...v);
  return v.map((x) => x / norm);
});

const pct = (xs: number[], p: number) =>
  +[...xs].sort((a, b) => a - b)[Math.min(xs.length - 1, Math.floor(p * xs.length))].toFixed(1);
function timeSearch(filtered: boolean): { p50: number; p95: number } {
  const ms = queries.map((q) => {
    const t = performance.now();
    searchVectorized(db, q, 20, 2, '_text', filtered ? { since } : undefined);
    return performance.now() - t;
  });
  return { p50: pct(ms, 0.5), p95: pct(ms, 0.95) };
}
const counts = () => ({
  vec: (db.prepare('SELECT COUNT(*) AS n FROM conv_vec_text').get() as { n: number }).n,
  map: (db.prepare('SELECT COUNT(*) AS n FROM conv_vec_map_text').get() as { n: number }).n,
});

const before = { ...counts(), knn: timeSearch(false), filtered: timeSearch(true) };
const t0 = performance.now();
const result = maintainVectors(db);
const maintainSecs = +((performance.now() - t0) / 1000).toFixed(1);
const after = {
  ...counts(),
  deadShare: +deadVectorShare(db, 'conv_vec_text').toFixed(3),
  knn: timeSearch(false),
  filtered: timeSearch(true),
};
console.log(JSON.stringify({ dims, queries: n, before, result, maintainSecs, after }, null, 1));
db.close();
