/**
 * Hybrid search engine: BM25 (FTS5) + vectors (sqlite-vec) + RRF + reranker.
 */

import type Database from 'better-sqlite3';
import type { MatchType, SearchContext, SearchOptions, SearchResult } from './models.js';
import { incrementStat, setStat } from './db.js';
import { RRF_K } from './constants.js';
import { logger } from './logger.js';
import { vectorIndexMatches } from './migration.js';

/** Upper bound on OR terms in one keyword expression (a safety limit, not pruning). */
export const MAX_QUERY_TERMS = 128;

/**
 * Turn user text into a safe FTS5 expression: whitespace-separated terms, each quoted as a
 * phrase (inner quotes doubled), joined with OR. Quoting makes punctuation, operator words
 * (AND, OR, NOT, NEAR) and stray quotes literal, so the expression can never be a syntax
 * error; a dotted or hyphenated name stays one phrase ("test_config.py" matches the tokens
 * test config py in order). Terms with no letter or digit are dropped. Returns '' when
 * nothing searchable remains.
 */
export function buildFtsQuery(
  text: string,
  opts: { prefix?: boolean; minChars?: number } = {},
): string {
  const minChars = opts.minChars ?? 1;
  const terms = text
    .split(/\s+/u)
    .filter((t) => (t.match(/[\p{L}\p{N}]/gu)?.length ?? 0) >= minChars)
    .slice(0, MAX_QUERY_TERMS);
  const star = opts.prefix ? '*' : '';
  return terms.map((t) => `"${t.replace(/"/g, '""')}"${star}`).join(' OR ');
}

/** Keyword-search failures by cause, since process start (reported by m9k_info). */
export const ftsErrorCounts = { syntax: 0, busy: 0, other: 0 };

function recordFtsError(err: unknown, expr: string): void {
  const code = (err as { code?: string }).code ?? '';
  const message = err instanceof Error ? err.message : String(err);
  const kind =
    code === 'SQLITE_BUSY' || code === 'SQLITE_LOCKED'
      ? 'busy'
      : /fts5|syntax|unterminated|no such column/i.test(message)
        ? 'syntax'
        : 'other';
  ftsErrorCounts[kind]++;
  logger.warn('search', `Keyword search failed (${kind}): ${message} — ${expr.slice(0, 80)}`);
}

/** Scope applied inside every channel, before its candidate cut. Dates are ISO-8601 UTC. */
export interface SearchFilters {
  project?: string;
  since?: string;
  until?: string;
}

function hasFilters(f?: SearchFilters): f is SearchFilters {
  return !!(f && (f.project || f.since || f.until));
}

/** SQL conditions on aliases c (chunk) and s (session), plus their parameters. */
function filterSql(f?: SearchFilters): { sql: string; params: string[] } {
  const parts: string[] = [];
  const params: string[] = [];
  if (f?.project) {
    parts.push('s.project = ?');
    params.push(f.project);
  }
  if (f?.since) {
    parts.push('c.timestamp >= ?');
    params.push(f.since);
  }
  if (f?.until) {
    parts.push('c.timestamp < ?');
    params.push(f.until);
  }
  return { sql: parts.map((p) => ` AND ${p}`).join(''), params };
}

/** BM25 over an FTS5 expression (build it with buildFtsQuery). Never throws. */
export function searchBM25(
  db: Database.Database,
  expr: string,
  limit: number,
  filters?: SearchFilters,
): SearchResult[] {
  if (!expr.trim()) return [];

  const f = filterSql(filters);
  const stmt = db.prepare(`
    SELECT c.id, c.session_id, c.timestamp, c.user_content, c.assistant_content,
           s.project, bm25(conv_chunks_fts) AS score
    FROM conv_chunks_fts
    JOIN conv_chunks c ON conv_chunks_fts.rowid = c.rowid
    JOIN conv_sessions s ON c.session_id = s.id
    WHERE conv_chunks_fts MATCH ?
      AND c.deleted_at IS NULL${f.sql}
    ORDER BY score
    LIMIT ?
  `);

  let rows: Array<{
    id: string;
    session_id: string;
    timestamp: string;
    user_content: string;
    assistant_content: string;
    project: string;
    score: number;
  }>;

  try {
    rows = stmt.all(expr, ...f.params, limit) as typeof rows;
  } catch (err) {
    recordFtsError(err, expr);
    return [];
  }

  return rows.map((row) => ({
    chunkId: row.id,
    snippet: (row.user_content + ' ' + row.assistant_content).slice(0, 100),
    // SQLite's bm25() is lower-is-better; every score in this module is higher-is-better.
    score: -row.score,
    project: row.project,
    timestamp: row.timestamp,
    matchType: 'bm25' as const,
    sessionId: row.session_id,
  }));
}

/**
 * Max L2 distance for vector results. For unit-normalized vectors:
 * L2 = sqrt(2 - 2*cos_sim), so maxDistance=1.3 ≈ cos_sim > 0.155
 * This filters out clearly irrelevant results at scale.
 */
const MAX_VECTOR_DISTANCE = 1.3;

export function searchVectorized(
  db: Database.Database,
  queryEmbedding: Float32Array,
  limit: number,
  maxDistance = MAX_VECTOR_DISTANCE,
  suffix = '_text',
  filters?: SearchFilters,
): SearchResult[] {
  if (hasFilters(filters)) {
    return searchVectorizedScoped(db, queryEmbedding, limit, maxDistance, suffix, filters);
  }

  // Two-step approach: sqlite-vec virtual tables are slow when JOINed directly
  // with regular tables (SQLite optimizer can't push down the MATCH efficiently).
  // Step 1: KNN search on the vec table only (~2ms for 5k vectors)
  const vecRows = db
    .prepare(
      `SELECT rowid, distance FROM conv_vec${suffix}
       WHERE embedding MATCH ? AND k = ?
       ORDER BY distance`,
    )
    .all(Buffer.from(queryEmbedding.buffer), limit) as Array<{
    rowid: number;
    distance: number;
  }>;

  if (vecRows.length === 0) return [];

  // Step 2: resolve rowids → chunk_ids via map table, then hydrate from chunks
  const rowids = vecRows.map((r) => r.rowid);
  const distanceByRowid = new Map(vecRows.map((r) => [r.rowid, r.distance]));

  const placeholders = rowids.map(() => '?').join(',');
  const mapRows = db
    .prepare(
      `SELECT vec_rowid, chunk_id FROM conv_vec_map${suffix}
       WHERE vec_rowid IN (${placeholders})`,
    )
    .all(...rowids) as Array<{ vec_rowid: number; chunk_id: string }>;

  if (mapRows.length === 0) return [];

  const chunkIds = mapRows.map((m) => m.chunk_id);
  const distanceByChunkId = new Map(
    mapRows.map((m) => [m.chunk_id, distanceByRowid.get(m.vec_rowid)!]),
  );

  const chunkPlaceholders = chunkIds.map(() => '?').join(',');
  const chunkRows = db
    .prepare(
      `SELECT c.id, c.user_content, c.assistant_content,
              c.session_id, c.timestamp, s.project
       FROM conv_chunks c
       JOIN conv_sessions s ON c.session_id = s.id
       WHERE c.id IN (${chunkPlaceholders})
         AND c.deleted_at IS NULL`,
    )
    .all(...chunkIds) as Array<{
    id: string;
    user_content: string;
    assistant_content: string;
    session_id: string;
    timestamp: string;
    project: string;
  }>;

  return chunkRows
    .map((row) => {
      const distance = distanceByChunkId.get(row.id)!;
      return {
        chunkId: row.id,
        snippet: (row.user_content + ' ' + row.assistant_content).slice(0, 100),
        score: 1 - distance,
        project: row.project,
        timestamp: row.timestamp,
        matchType: (suffix === '_code' ? 'vector_code' : 'vector_text') as MatchType,
        sessionId: row.session_id,
      };
    })
    .filter((r) => 1 - r.score <= maxDistance)
    .sort((a, b) => b.score - a.score);
}

/**
 * Filtered vector search, exact: every vector is scored and the scope is applied before
 * the top-k cut, so a match inside the scope is found however far down the global
 * nearest-neighbour list it sits. The scan-first join order keeps the cost close to an
 * unfiltered KNN query (vec0 is itself a flat scan).
 */
function searchVectorizedScoped(
  db: Database.Database,
  queryEmbedding: Float32Array,
  limit: number,
  maxDistance: number,
  suffix: string,
  filters: SearchFilters,
): SearchResult[] {
  const f = filterSql(filters);
  const rows = db
    .prepare(
      `SELECT c.id, c.user_content, c.assistant_content, c.session_id, c.timestamp, s.project,
              v.distance
       FROM (SELECT rowid AS vec_rowid, vec_distance_l2(embedding, ?) AS distance
             FROM conv_vec${suffix}) v
       CROSS JOIN conv_vec_map${suffix} m ON m.vec_rowid = v.vec_rowid
       CROSS JOIN conv_chunks c ON c.id = m.chunk_id
       CROSS JOIN conv_sessions s ON s.id = c.session_id
       WHERE c.deleted_at IS NULL AND v.distance <= ?${f.sql}
       ORDER BY v.distance
       LIMIT ?`,
    )
    .all(Buffer.from(queryEmbedding.buffer), maxDistance, ...f.params, limit) as Array<{
    id: string;
    user_content: string;
    assistant_content: string;
    session_id: string;
    timestamp: string;
    project: string;
    distance: number;
  }>;
  return rows.map((row) => ({
    chunkId: row.id,
    snippet: (row.user_content + ' ' + row.assistant_content).slice(0, 100),
    score: 1 - row.distance,
    project: row.project,
    timestamp: row.timestamp,
    matchType: (suffix === '_code' ? 'vector_code' : 'vector_text') as MatchType,
    sessionId: row.session_id,
  }));
}

/**
 * Reciprocal Rank Fusion: combines N ranked lists.
 * score(doc) = sum over lists of 1/(K + rank_in_list)
 * matchType is 'hybrid' when a doc appears in 2+ lists.
 */
export function fusionRRF(lists: SearchResult[][], k = RRF_K): SearchResult[] {
  const scores = new Map<string, { score: number; result: SearchResult; sourceCount: number }>();

  for (const list of lists) {
    for (let i = 0; i < list.length; i++) {
      const r = list[i];
      const rrfScore = 1 / (k + i + 1); // rank is 1-based
      const existing = scores.get(r.chunkId);
      if (existing) {
        existing.score += rrfScore;
        existing.sourceCount++;
        existing.result.matchType = 'hybrid';
      } else {
        scores.set(r.chunkId, {
          score: rrfScore,
          result: { ...r },
          sourceCount: 1,
        });
      }
    }
  }

  return Array.from(scores.values())
    .sort((a, b) => b.score - a.score)
    .map(({ score, result }) => ({ ...result, score }));
}

/**
 * Project boost factor: results from currentProject get a score boost.
 * Applied as a multiplier on the fused (or reranked rank) score, which is always positive.
 * Moderate enough to promote same-project results without hiding cross-project ones.
 */
const PROJECT_BOOST_FACTOR = 1.5;
const SESSION_BOOST_FACTOR = 1.2;

/**
 * Collapse results whose exchange text is identical (same hash), keeping the best-ranked
 * copy and recording the other sessions on it, so copies do not crowd out distinct hits.
 */
function dedupeByHash(db: Database.Database, results: SearchResult[]): SearchResult[] {
  if (results.length < 2) return results;
  const hashes = new Map<string, string>();
  for (let i = 0; i < results.length; i += 500) {
    const ids = results.slice(i, i + 500).map((r) => r.chunkId);
    const rows = db
      .prepare(`SELECT id, hash FROM conv_chunks WHERE id IN (${ids.map(() => '?').join(',')})`)
      .all(...ids) as Array<{ id: string; hash: string }>;
    for (const row of rows) hashes.set(row.id, row.hash);
  }
  const kept = new Map<string, SearchResult>();
  const out: SearchResult[] = [];
  for (const r of results) {
    const h = hashes.get(r.chunkId);
    const first = h ? kept.get(h) : undefined;
    if (!first) {
      if (h) kept.set(h, r);
      out.push(r);
    } else if (r.sessionId !== first.sessionId && !first.alsoIn?.includes(r.sessionId)) {
      (first.alsoIn ??= []).push(r.sessionId);
    }
  }
  return out;
}

/** At most `max` results per session, in order; later results refill the freed slots. */
function capPerSession(results: SearchResult[], max?: number): SearchResult[] {
  if (!max || max < 1) return results;
  const perSession = new Map<string, number>();
  return results.filter((r) => {
    const n = perSession.get(r.sessionId) ?? 0;
    perSession.set(r.sessionId, n + 1);
    return n < max;
  });
}

/** Fused candidates the cross-encoder scores per search (serially, in q8), whatever the limit. */
export const RERANK_POOL = 20;

/** Score for position i of a final ranking, on the same scale as one RRF channel. */
function rankScore(i: number): number {
  return 1 / (RRF_K + i + 1);
}

export async function search(
  db: Database.Database,
  options: SearchOptions,
  ctx?: SearchContext,
): Promise<SearchResult[]> {
  if (!options.query.trim()) return [];

  const t0 = performance.now();
  const autoFuzzyThreshold = ctx?.autoFuzzyThreshold ?? 3;

  // Every channel applies the scope itself, before its own cut.
  const filters: SearchFilters = {
    project: options.project,
    since: options.since,
    until: options.until,
  };

  // BM25 search (always available)
  const bm25Results = searchBM25(db, buildFtsQuery(options.query), options.limit * 2, filters);
  const t1 = performance.now();
  logger.debug('search', ` BM25: ${(t1 - t0).toFixed(0)}ms (${bm25Results.length} results)`);

  let results: SearchResult[];

  // Triple vector search: BM25 + text vectors + code vectors, fused via RRF
  const vecLists: SearchResult[][] = [bm25Results];

  // A generation built by another model or recipe (or mid-migration) is skipped:
  // keyword search keeps working, and vectors return once the new index is published.
  if (
    ctx?.vecTextEnabled &&
    ctx.embedderText &&
    vectorIndexMatches(db, ctx.embedderText, '_text')
  ) {
    try {
      const t2 = performance.now();
      const embedFn = ctx.embedderText.embedQuery ?? ctx.embedderText.embed;
      const emb = await embedFn.call(ctx.embedderText, options.query);
      const t3 = performance.now();
      logger.debug('search', ` text embed: ${(t3 - t2).toFixed(0)}ms`);
      vecLists.push(
        searchVectorized(db, emb, options.limit * 2, MAX_VECTOR_DISTANCE, '_text', filters),
      );
      logger.debug('search', ` text vec search: ${(performance.now() - t3).toFixed(0)}ms`);
    } catch (err) {
      logger.warn('search', 'Text vector search failed — degrading gracefully', err);
    }
  }

  if (
    ctx?.vecCodeEnabled &&
    ctx.embedderCode &&
    vectorIndexMatches(db, ctx.embedderCode, '_code')
  ) {
    try {
      const t4 = performance.now();
      const embedFn = ctx.embedderCode.embedQuery ?? ctx.embedderCode.embed;
      const emb = await embedFn.call(ctx.embedderCode, options.query);
      const t5 = performance.now();
      logger.debug('search', ` code embed: ${(t5 - t4).toFixed(0)}ms`);
      vecLists.push(
        searchVectorized(db, emb, options.limit * 2, MAX_VECTOR_DISTANCE, '_code', filters),
      );
      logger.debug('search', ` code vec search: ${(performance.now() - t5).toFixed(0)}ms`);
    } catch (err) {
      logger.warn('search', 'Code vector search failed — degrading gracefully', err);
    }
  }

  // Every channel goes through RRF, even alone, so scores share one direction and scale.
  results = fusionRRF(vecLists);
  logger.debug('search', ` RRF: ${(performance.now() - t1).toFixed(0)}ms total`);

  // Auto-fuzzy fallback: too few results -> prefix search, fused as its own channel
  if (results.length < autoFuzzyThreshold) {
    const fuzzyResults = searchFuzzy(db, options.query, options.limit * 2, filters);
    if (fuzzyResults.length > 0) {
      // Prefix matches re-rank, but don't relabel what the other channels already found.
      const types = new Map(fusionRRF(vecLists).map((r) => [r.chunkId, r.matchType]));
      results = fusionRRF([...vecLists, fuzzyResults]).map((r) =>
        types.has(r.chunkId) ? { ...r, matchType: types.get(r.chunkId)! } : r,
      );
    }
  }

  // Copies of one exchange (sessions resumed or forked from another) take one slot.
  results = dedupeByHash(db, results);

  // Reranker: the cross-encoder reorders only the top RERANK_POOL fused candidates (its
  // cost grows with the pool, not with the output limit); the rest keep fused order
  // behind them. Logits become rank scores on the RRF scale, so later multipliers stay
  // well-defined.
  const tRerank0 = performance.now();
  if (ctx?.reranker && results.length > 1) {
    try {
      const pool = results.slice(0, RERANK_POOL);
      const ph = pool.map(() => '?').join(',');
      const texts = new Map(
        (
          db
            .prepare(
              `SELECT id, user_content || ' ' || assistant_content AS content
               FROM conv_chunks WHERE id IN (${ph})`,
            )
            .all(...pool.map((r) => r.chunkId)) as Array<{ id: string; content: string }>
        ).map((row) => [row.id, row.content]),
      );
      const documents = pool.map((r) => ({
        id: r.chunkId,
        content: (texts.get(r.chunkId) ?? r.snippet).slice(0, 512),
      }));

      const reranked = await ctx.reranker.rerank(options.query, documents, pool.length);
      const byId = new Map(pool.map((r) => [r.chunkId, r]));
      results = [
        ...reranked.filter((rr) => byId.has(rr.id)).map((rr) => byId.get(rr.id)!),
        ...results.slice(RERANK_POOL),
      ].map((r, i) => ({ ...r, score: rankScore(i) }));
    } catch (err) {
      logger.warn('search', 'Reranker failed — keeping original order', err);
    }
  }
  logger.debug('search', ` reranker: ${(performance.now() - tRerank0).toFixed(0)}ms`);

  // Affinity boosts multiply the (positive) fused or rank score, before the final cut.
  const projectBoost = !options.project && options.currentProject ? PROJECT_BOOST_FACTOR : 1;
  const sessionBoost = options.currentSession ? SESSION_BOOST_FACTOR : 1;
  if (projectBoost !== 1 || sessionBoost !== 1) {
    for (const r of results) {
      if (projectBoost !== 1 && r.project === options.currentProject) r.score *= projectBoost;
      if (sessionBoost !== 1 && r.sessionId === options.currentSession) r.score *= sessionBoost;
    }
    results.sort((a, b) => b.score - a.score);
  }

  // Re-sort if order is date-based (default is score from BM25/RRF/reranker)
  if (options.order === 'date_asc') {
    results.sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  } else if (options.order === 'date_desc') {
    results.sort((a, b) => b.timestamp.localeCompare(a.timestamp));
  }

  const finalResults = capPerSession(results, options.maxPerSession).slice(0, options.limit);

  // Update usage counters (best-effort, don't fail the search)
  try {
    incrementStat(db, 'search_count');
    if (finalResults.length > 0) {
      incrementStat(db, 'hit_count');
      const tokensServed = finalResults.reduce(
        (sum, r) => sum + Math.ceil(r.snippet.length / 4),
        0,
      );
      incrementStat(db, 'tokens_served', tokensServed);
    }
    setStat(db, 'last_search_at', new Date().toISOString());
  } catch {
    // Non-critical — don't fail searches if stats write fails
  }

  return finalResults;
}

/**
 * Prefix fallback ("fuzzy" here means prefix expansion, not typo correction): each term
 * of 3+ letters or digits becomes a quoted prefix phrase ("term"*), joined with OR.
 */
export function searchFuzzy(
  db: Database.Database,
  query: string,
  limit: number,
  filters?: SearchFilters,
): SearchResult[] {
  const expr = buildFtsQuery(query, { prefix: true, minChars: 3 });
  if (!expr) return [];
  const results = searchBM25(db, expr, limit, filters);
  return results.map((r) => ({ ...r, matchType: 'fuzzy' as const }));
}
