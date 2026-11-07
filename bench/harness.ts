/**
 * Real-code search harness: runs a labeled query set through the actual search() and tool
 * handlers against a database copy, and writes one JSON line per query with per-stage outputs.
 *
 * Usage:
 *   tsx bench/harness.ts --db <copy.db> --queries <set.jsonl> --out <dir> [--set main|hard|errors|files]
 *                        [--model <registry key>] [--no-rerank] [--limit 10] [--max-per-session N]
 *                        [--depth N]   (minimum candidates per channel; unset keeps limit × 2)
 *
 * Model files are read from M9K_MODEL_CACHE (a transformers.js cache dir) when set; remote downloads
 * stay off. Boosts are neutral (no currentProject / currentSession). Nothing here writes to the DB
 * unless the code under test does (stats counters), so pass a throwaway copy.
 */

import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
import { env } from '@huggingface/transformers';
import { openDatabase } from '../src/db.js';
import { resolveDateRange } from '../src/dates.js';
import * as S from '../src/search.js';
import { TransformersJsEmbedder } from '../src/embedder.js';
import { TransformersJsReranker } from '../src/reranker.js';
import { lookupModelSpec, DEFAULT_TEXT_MODEL_KEY } from '../src/constants.js';
import { registerSpecializedTools } from '../src/tools/specialized.js';
import type { Embedder, SearchContext, SearchResult } from '../src/models.js';
import type { ToolContext } from '../src/tools/context.js';

interface Query {
  id: string;
  query: string;
  /** When a logged query was asked; relative dates resolve against it. */
  asked_at?: string;
  timeframe?: string | null;
  opts: {
    since?: string;
    until?: string;
    project?: string;
    order?: 'score' | 'date_asc' | 'date_desc';
    asked_at?: string;
  };
}

const argv = process.argv.slice(2);
const arg = (name: string, dflt?: string): string | undefined => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : dflt;
};
const flag = (name: string): boolean => argv.includes(`--${name}`);

const dbPath = arg('db')!;
const queriesPath = arg('queries')!;
const outDir = arg('out')!;
const set = arg('set', 'main')!;
const modelKey = arg('model', DEFAULT_TEXT_MODEL_KEY)!;
const LIMIT = parseInt(arg('limit', '10')!, 10);
const MAX_PER_SESSION = arg('max-per-session') ? parseInt(arg('max-per-session')!, 10) : undefined;
const DEPTH_ARG = arg('depth') ? parseInt(arg('depth')!, 10) : undefined;
const DEPTH = S.channelDepth(LIMIT, DEPTH_ARG);
if (!dbPath || !queriesPath || !outDir) {
  console.error('usage: tsx bench/harness.ts --db <copy.db> --queries <set.jsonl> --out <dir>');
  process.exit(2);
}

// db.ts loads sqlite-vec through require(), which the esbuild bundle shims; do the same here.
(globalThis as { require?: NodeRequire }).require = createRequire(import.meta.url);

if (process.env.M9K_MODEL_CACHE) env.cacheDir = process.env.M9K_MODEL_CACHE;
env.allowRemoteModels = false;

// openDatabase runs the real schema upgrade on the copy, as the daemon would.
const { db, vecEnabled } = openDatabase(dbPath);
if (!vecEnabled) throw new Error('sqlite-vec failed to load');

const spec = lookupModelSpec(modelKey);
if (!spec) throw new Error(`unknown model key ${modelKey}`);
const embedder: Embedder = new TransformersJsEmbedder(spec);
const reranker = flag('no-rerank') ? null : new TransformersJsReranker();

const baseCtx: SearchContext = {
  embedderText: embedder,
  embedderCode: null,
  reranker: null,
  vecTextEnabled: true,
  vecCodeEnabled: false,
  autoFuzzyThreshold: 3,
};

const ids = (rs: SearchResult[]): string[] => rs.map((r) => r.chunkId);

function applyFilters(rs: SearchResult[], o: Query['opts']): SearchResult[] {
  return rs.filter(
    (r) =>
      (!o.project || r.project === o.project) &&
      (!o.since || r.timestamp >= o.since) &&
      (!o.until || r.timestamp < o.until),
  );
}

/** True when the raw string is not a valid FTS5 expression (what 1.0.2 sends to MATCH). */
function rawFtsError(q: string): boolean {
  try {
    db.prepare('SELECT 1 FROM conv_chunks_fts WHERE conv_chunks_fts MATCH ? LIMIT 1').get(q);
    return false;
  } catch {
    return true;
  }
}

/** Capture tool handlers without an MCP transport. */
function toolHandlers(ctx: SearchContext): Record<string, (a: unknown) => Promise<unknown>> {
  const tools: Record<string, (a: unknown) => Promise<unknown>> = {};
  const fake = {
    registerTool: (name: string, _def: unknown, handler: (a: unknown) => Promise<unknown>) => {
      tools[name] = handler;
    },
  };
  const toolCtx = {
    db,
    searchContext: ctx,
    cfg: { candidateDepth: DEPTH_ARG },
  } as unknown as ToolContext;
  registerSpecializedTools(fake as never, toolCtx);
  return tools;
}

function parseToolText(res: unknown): Array<{ chunkId: string }> {
  const r = res as { content: Array<{ text: string }> };
  return JSON.parse(r.content[0].text) as Array<{ chunkId: string }>;
}

async function main(): Promise<void> {
  fs.mkdirSync(outDir, { recursive: true });
  const queries = fs
    .readFileSync(queriesPath, 'utf8')
    .trim()
    .split('\n')
    .map((l) => JSON.parse(l) as Query);
  const runsPath = path.join(outDir, `runs_${set}.jsonl`);
  const out = fs.createWriteStream(runsPath);
  const fullCtx: SearchContext = { ...baseCtx, reranker };
  const tools = toolHandlers(fullCtx);
  const latencies: number[] = [];

  for (const q of queries) {
    // Dates go through the same normalisation as the m9k_search tool; a rejected date is
    // recorded as that query's tool error (empty result), as the tool would return it.
    let o: Query['opts'];
    try {
      const askedAt = q.asked_at ?? q.opts.asked_at;
      const range = resolveDateRange(
        { since: q.opts.since, until: q.opts.until, timeframe: q.timeframe ?? undefined },
        askedAt ? new Date(askedAt) : new Date(),
      );
      o = { project: q.opts.project, order: q.opts.order, ...range };
    } catch (err) {
      out.write(
        JSON.stringify({
          id: q.id,
          dateError: String(err),
          runs: { full: [], full_norerank: [], bm25: [], vector: [], fuzzy: [] },
        }) + '\n',
      );
      continue;
    }
    const syntaxBefore = S.ftsErrorCounts?.syntax ?? 0;
    const rec: Record<string, unknown> = { id: q.id, ftsError: rawFtsError(q.query) };
    const runs: Record<string, string[]> = {};

    if (set === 'files') {
      const t0 = performance.now();
      const res = await tools['m9k_file_history']({ filePath: q.query, limit: LIMIT });
      latencies.push(performance.now() - t0);
      runs.file_history = parseToolText(res).map((r) => r.chunkId);
    } else if (set === 'errors') {
      const t0 = performance.now();
      const res = await tools['m9k_errors']({ errorMessage: q.query, limit: LIMIT });
      latencies.push(performance.now() - t0);
      runs.errors_tool = parseToolText(res).map((r) => r.chunkId);
    }

    if (set !== 'files') {
      const emb = await (embedder.embedQuery ?? embedder.embed).call(embedder, q.query);
      const expr = S.buildFtsQuery ? S.buildFtsQuery(q.query) : q.query;
      const bm25raw = S.searchBM25(db, expr, DEPTH);
      const vecraw = S.searchVectorized(db, emb, DEPTH);
      const fuzzyraw = S.searchFuzzy(db, q.query, DEPTH);
      runs.bm25 = ids(applyFilters(bm25raw, o)).slice(0, LIMIT);
      runs.vector = ids(applyFilters(vecraw, o)).slice(0, LIMIT);
      runs.fuzzy = ids(applyFilters(fuzzyraw, o)).slice(0, LIMIT);
      const opts = {
        query: q.query,
        limit: LIMIT,
        ...o,
        maxPerSession: MAX_PER_SESSION,
        candidateDepth: DEPTH_ARG,
      };
      runs.full_norerank = ids(await S.search(db, opts, baseCtx));
      const t0 = performance.now();
      const full = await S.search(db, opts, fullCtx);
      if (set !== 'errors') latencies.push(performance.now() - t0);
      runs.full = ids(full);
      rec.stages = {
        bm25_raw: ids(bm25raw),
        vector_raw: ids(vecraw),
        fuzzy_raw: ids(fuzzyraw),
        fused: ids(S.fusionRRF([bm25raw, vecraw])),
      };
      rec.full_detail = full.map((r) => ({
        id: r.chunkId,
        mt: r.matchType,
        s: +r.score.toFixed(3),
      }));
    }
    rec.bm25SyntaxErrors = (S.ftsErrorCounts?.syntax ?? 0) - syntaxBefore;
    rec.runs = runs;
    out.write(JSON.stringify(rec) + '\n');
  }
  await new Promise<void>((resolve) => out.end(resolve));

  latencies.sort((a, b) => a - b);
  const pct = (p: number): number =>
    latencies[Math.min(latencies.length - 1, Math.floor(p * latencies.length))];
  const summary = {
    set,
    model: modelKey,
    rerank: !!reranker,
    depth: DEPTH,
    n: queries.length,
    p50_ms: +pct(0.5).toFixed(1),
    p95_ms: +pct(0.95).toFixed(1),
  };
  fs.writeFileSync(path.join(outDir, `summary_${set}.json`), JSON.stringify(summary, null, 1));
  console.log(JSON.stringify(summary));
}

main().then(
  () => db.close(),
  (err) => {
    console.error(err);
    process.exit(1);
  },
);
