/**
 * Optional local log of tool calls, off by default (config `searchLog: true`, or M9K_SEARCH_LOG=true).
 *
 * One JSON line per call in ~/.melchizedek/logs/search.jsonl: time, tool, the caller's session and
 * project, the query arguments, duration, whether it failed, the result count, the first 10 result
 * IDs and, for searches, the time spent in each stage. m9k_context and m9k_full calls carry the
 * chunk or session ID they open, so whether a search result was then opened can be read from the
 * log. The file is rotated once, to search.jsonl.1, past 5 MB. Logging never fails a tool call.
 */

import { AsyncLocalStorage } from 'node:async_hooks';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';

const stages = new AsyncLocalStorage<Record<string, number>>();

/** Record how long a search stage took (ms) for the tool call in progress, when it is logged. */
export function recordStage(name: string, ms: number): void {
  const s = stages.getStore();
  if (s) s[name] = Math.round(ms);
}

const ARG_KEYS = [
  'query',
  'errorMessage',
  'filePath',
  'chunkId',
  'sessionId',
  'project',
  'since',
  'until',
  'timeframe',
  'order',
  'limit',
];

export interface ToolLogOptions {
  file?: string;
  maxBytes?: number;
  session?: string;
  project?: string;
}

type Handler = (...a: unknown[]) => unknown;
type ToolResult = { content?: Array<{ text?: string }>; isError?: boolean };

/** Result count and first IDs, when the tool's text is a JSON list (or holds `results`). */
function summarize(result: unknown): { n: number | null; ids: string[] } {
  try {
    const text = (result as ToolResult)?.content?.[0]?.text;
    const parsed: unknown = text ? JSON.parse(text) : null;
    const items = Array.isArray(parsed)
      ? parsed
      : Array.isArray((parsed as { results?: unknown })?.results)
        ? (parsed as { results: unknown[] }).results
        : null;
    if (!items) return { n: null, ids: [] };
    const ids = items
      .slice(0, 10)
      .map(
        (r) =>
          (r as { chunkId?: string; sessionId?: string }).chunkId ??
          (r as { sessionId?: string }).sessionId,
      )
      .filter((x): x is string => typeof x === 'string');
    return { n: items.length, ids };
  } catch {
    return { n: null, ids: [] };
  }
}

function append(file: string, maxBytes: number, entry: object): void {
  try {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    if (fs.existsSync(file) && fs.statSync(file).size > maxBytes) fs.renameSync(file, `${file}.1`);
    fs.appendFileSync(file, JSON.stringify(entry) + '\n');
  } catch {
    /* a log write must never fail the tool call */
  }
}

/** Wrap every tool registered on `server` from now on so each call is logged. */
export function logToolCalls(server: McpServer, opts: ToolLogOptions = {}): void {
  const file = opts.file ?? path.join(os.homedir(), '.melchizedek', 'logs', 'search.jsonl');
  const maxBytes = opts.maxBytes ?? 5 * 1024 * 1024;
  const target = server as unknown as { registerTool: (...a: unknown[]) => unknown };
  const register = target.registerTool.bind(server);
  target.registerTool = (name: unknown, config: unknown, cb: unknown) =>
    register(name, config, async (...a: unknown[]) => {
      // Tools with an input schema get (args, extra); the rest get (extra) only.
      const hasArgs = !!(config as { inputSchema?: unknown })?.inputSchema;
      const args = (hasArgs ? a[0] : {}) as Record<string, unknown>;
      const st: Record<string, number> = {};
      const t0 = performance.now();
      let result: unknown;
      let failed = false;
      try {
        result = await stages.run(st, () => (cb as Handler)(...a));
        return result;
      } catch (err) {
        failed = true;
        throw err;
      } finally {
        const picked = Object.fromEntries(
          ARG_KEYS.filter((k) => args?.[k] !== undefined).map((k) => [k, args[k]]),
        );
        append(file, maxBytes, {
          ts: new Date().toISOString(),
          tool: name,
          session: opts.session ?? null,
          project: opts.project ?? null,
          args: picked,
          ms: Math.round(performance.now() - t0),
          error: failed || !!(result as ToolResult)?.isError,
          ...summarize(result),
          ...(Object.keys(st).length ? { stages: st } : {}),
        });
      }
    });
}
