/**
 * Specialized search tools: m9k_file_history, m9k_errors, m9k_similar_work.
 */

import path from 'path';
import { z } from 'zod';
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import type Database from 'better-sqlite3';
import { search, searchBM25 } from '../search.js';
import { DateInputError, dateErrorResult, resolveDateRange } from '../dates.js';
import { SOURCE_PARAM, type ToolContext } from './context.js';

export function registerSpecializedTools(server: McpServer, ctx: ToolContext): void {
  server.registerTool(
    'm9k_file_history',
    {
      description:
        'Find past conversations that touched a specific file. Searches metadata (tool_use file_path) and text content.',
      inputSchema: {
        filePath: z.string().describe('File path to search for (e.g. "src/server.ts")'),
        limit: z.number().int().min(1).max(50).default(10).describe('Max results'),
        source: SOURCE_PARAM,
      },
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ filePath, limit }) => ({
      content: [
        { type: 'text' as const, text: JSON.stringify(fileHistory(ctx.db, filePath, limit)) },
      ],
    }),
  );

  server.registerTool(
    'm9k_errors',
    {
      description:
        'Find past solutions for an error message. Returns error context + how it was resolved.',
      inputSchema: {
        errorMessage: z.string().describe('The error message or keywords from the error'),
        limit: z.number().int().min(1).max(20).default(5).describe('Max results'),
        source: SOURCE_PARAM,
      },
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ errorMessage, limit }) => {
      // No cross-encoder here: it is trained on natural-language questions and reorders
      // error strings worse than fusion does (top-5 session hits 18 of 25 without it, 15 with).
      const results = await search(
        ctx.db,
        { query: errorMessage, limit: limit * 3 },
        { ...ctx.searchContext, reranker: null },
      );

      // Enrich with error/solution extraction, filter by substantial assistant response
      const enriched: Array<{
        chunkId: string;
        sessionId: string;
        project: string;
        timestamp: string;
        error: string;
        solution: string;
        matchType: string;
      }> = [];

      for (const r of results) {
        const chunk = ctx.db
          .prepare('SELECT user_content, assistant_content FROM conv_chunks WHERE id = ?')
          .get(r.chunkId) as { user_content: string; assistant_content: string } | undefined;

        if (!chunk || chunk.assistant_content.length < 20) continue;

        enriched.push({
          chunkId: r.chunkId,
          sessionId: r.sessionId,
          project: r.project,
          timestamp: r.timestamp,
          error: chunk.user_content.slice(0, 200),
          solution: chunk.assistant_content.slice(0, 300),
          matchType: r.matchType,
        });

        if (enriched.length >= limit) break;
      }

      return {
        content: [{ type: 'text' as const, text: JSON.stringify(enriched) }],
      };
    },
  );

  server.registerTool(
    'm9k_similar_work',
    {
      description:
        "Find past work similar to what you're about to do. Use at the start of a complex task to see previous approaches. Unlike m9k_search, this prioritizes chunks with rich metadata (multiple tools used, multiple files touched).",
      inputSchema: {
        description: z.string().describe('Description of the current task'),
        limit: z.number().int().min(1).max(20).default(5),
        since: z
          .string()
          .optional()
          .describe(
            "Only results from this date on: ISO-8601 (local time unless it has an offset) or relative ('yesterday', '3 days ago').",
          ),
        until: z
          .string()
          .optional()
          .describe('Only results before this date (exclusive). Same formats as since.'),
        timeframe: z
          .string()
          .optional()
          .describe(
            "Named date range in local time, e.g. 'today', 'yesterday', 'last week', 'this month', 'past 3 days', '2 hours ago' (one hour either side). since/until override its sides.",
          ),
        source: SOURCE_PARAM,
      },
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async ({ description, limit, since, until, timeframe }) => {
      let range;
      try {
        range = resolveDateRange({ since, until, timeframe }, ctx.clock?.() ?? new Date());
      } catch (err) {
        if (err instanceof DateInputError) return dateErrorResult(err);
        throw err;
      }
      // Use the same search pipeline as m9k_search
      const results = await search(
        ctx.db,
        { query: description, limit: limit * 3, since: range.since, until: range.until },
        ctx.searchContext,
      );

      // Apply metadata bonus scoring
      const enriched = results.map((r) => {
        const chunk = ctx.db
          .prepare('SELECT metadata_json FROM conv_chunks WHERE id = ?')
          .get(r.chunkId) as { metadata_json: string } | undefined;

        let metadataBonus = 0;
        if (chunk?.metadata_json) {
          try {
            const meta = JSON.parse(chunk.metadata_json) as {
              toolCalls?: string[];
              filePaths?: string[];
            };
            if (meta.toolCalls && meta.toolCalls.length >= 3) metadataBonus += 0.2;
            if (meta.filePaths && meta.filePaths.length >= 2) metadataBonus += 0.1;
          } catch {
            // ignore parse error
          }
        }

        return {
          chunkId: r.chunkId,
          snippet: r.snippet,
          // Multiplicative, so the bonus means the same whatever the score scale.
          score: r.score * (1 + metadataBonus),
          metadataBonus,
          project: r.project,
          timestamp: r.timestamp,
          matchType: r.matchType,
          sessionId: r.sessionId,
        };
      });

      // Re-sort by adjusted score, take top limit
      enriched.sort((a, b) => b.score - a.score);
      return {
        content: [{ type: 'text' as const, text: JSON.stringify(enriched.slice(0, limit)) }],
      };
    },
  );
}

type FileMatch = 'path' | 'suffix' | 'basename' | 'metadata' | 'text';

interface FileRow {
  id: string;
  session_id: string;
  user_content: string;
  assistant_content: string;
  timestamp: string;
  metadata_json: string;
  project: string;
}

const FILE_ROW_SQL = `SELECT c.id, c.session_id, c.user_content, c.assistant_content, c.timestamp,
                             c.metadata_json, s.project
                      FROM conv_chunks c JOIN conv_sessions s ON s.id = c.session_id`;

/** Escape LIKE wildcards and the escape character, for ESCAPE '\\'. */
function escapeLike(text: string): string {
  return text.replace(/[\\%_]/g, '\\$&');
}

/** The text as it appears inside a JSON string (metadata_json is written by JSON.stringify). */
function jsonEscaped(text: string): string {
  return JSON.stringify(text).slice(1, -1);
}

/**
 * Chunks that touched a file, best evidence first:
 * 1. recorded tool paths (metadata filePaths): the exact path, then a path ending in the
 *    given one (src/server.ts), then the same basename in another directory;
 * 2. the path anywhere in the metadata, as a literal substring (a directory, a pattern);
 * 3. the basename as one quoted text phrase (test_config.py -> "test config py"), for the
 *    remaining slots only.
 * Each tier is newest first. Nothing here can throw on user input.
 */
function fileHistory(db: Database.Database, filePath: string, limit: number) {
  const wanted = filePath.trim().replace(/^\.\//, '');
  const basename = path.basename(wanted);
  if (!basename) return [];
  const pool = limit * 2;
  const tail = '/' + wanted.replace(/^\/+/, '');
  const found = new Map<string, { row: FileRow; match: FileMatch }>();
  const add = (rows: FileRow[], match: (r: FileRow) => FileMatch) => {
    for (const r of rows) if (!found.has(r.id)) found.set(r.id, { row: r, match: match(r) });
  };

  try {
    const rows = db
      .prepare(
        `SELECT * FROM (
           SELECT c.id, c.session_id, c.user_content, c.assistant_content, c.timestamp,
                  c.metadata_json, s.project,
                  MIN(CASE WHEN fp.value = @wanted THEN 0
                           WHEN substr(fp.value, -length(@tail)) = @tail THEN 1
                           ELSE 2 END) AS tier
           FROM conv_chunks c
           JOIN conv_sessions s ON s.id = c.session_id
           JOIN json_each(c.metadata_json, '$.filePaths') fp
           WHERE c.deleted_at IS NULL
             AND c.metadata_json LIKE @prefilter ESCAPE '\\'
             AND json_valid(c.metadata_json)
             AND (fp.value = @wanted OR fp.value = @basename
                  OR substr(fp.value, -length(@baseTail)) = @baseTail)
           GROUP BY c.id)
         ORDER BY tier, timestamp DESC
         LIMIT @pool`,
      )
      .all({
        wanted,
        tail,
        basename,
        baseTail: '/' + basename,
        prefilter: `%${escapeLike(jsonEscaped(basename))}%`,
        pool,
      }) as Array<FileRow & { tier: number }>;
    const tiers: FileMatch[] = ['path', 'suffix', 'basename'];
    add(rows, (r) => tiers[(r as FileRow & { tier: number }).tier]);
  } catch {
    // malformed metadata in an old row: fall through to the literal and text tiers
  }

  if (found.size < limit) {
    const rows = db
      .prepare(
        `${FILE_ROW_SQL}
         WHERE c.deleted_at IS NULL AND c.metadata_json LIKE ? ESCAPE '\\'
         ORDER BY c.timestamp DESC
         LIMIT ?`,
      )
      .all(`%${escapeLike(jsonEscaped(wanted))}%`, pool) as FileRow[];
    add(rows, () => 'metadata');
  }

  if (found.size < limit && /[\p{L}\p{N}]/u.test(basename)) {
    const phrase = `"${basename.replace(/"/g, '""')}"`;
    const hits = searchBM25(db, phrase, pool, undefined, 'date_desc');
    if (hits.length > 0) {
      const ph = hits.map(() => '?').join(',');
      const rows = db
        .prepare(`${FILE_ROW_SQL} WHERE c.id IN (${ph})`)
        .all(...hits.map((h) => h.chunkId)) as FileRow[];
      const byId = new Map(rows.map((r) => [r.id, r]));
      add(
        hits.map((h) => byId.get(h.chunkId)).filter((r): r is FileRow => !!r),
        () => 'text',
      );
    }
  }

  return [...found.values()].slice(0, limit).map(({ row, match }) => {
    let filePaths: string[] = [];
    try {
      filePaths = (JSON.parse(row.metadata_json) as { filePaths?: string[] }).filePaths ?? [];
    } catch {
      // ignore parse error
    }
    return {
      chunkId: row.id,
      sessionId: row.session_id,
      project: row.project,
      timestamp: row.timestamp,
      snippet: (row.user_content + ' ' + row.assistant_content).slice(0, 150),
      filePaths,
      match,
    };
  });
}
