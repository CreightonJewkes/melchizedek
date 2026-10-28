import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type Database from 'better-sqlite3';
import { openMemoryDatabase } from '../src/db.js';
import { registerSearchTools } from '../src/tools/search.js';
import { captureTools } from './helpers/fixtures.js';

describe('m9k_full and m9k_context: order and size budgets (C6)', () => {
  let db: Database.Database;
  let call: (name: string, args: Record<string, unknown>) => Promise<unknown>;

  beforeEach(() => {
    db = openMemoryDatabase().db;
    db.prepare(
      "INSERT INTO conv_sessions (id, project, jsonl_path, started_at) VALUES ('s', '/p', '/t', '2026-01-01')",
    ).run();
    const ins = db.prepare(
      `INSERT INTO conv_chunks (id, session_id, idx, user_content, assistant_content, hash, timestamp)
       VALUES (?, 's', ?, ?, ?, ?, '2026-01-01T00:00:00Z')`,
    );
    // Inserted in id order; chunk 3 is large.
    for (let i = 0; i < 7; i++) {
      ins.run(`s:${i}`, i, `prompt ${i}`, i === 3 ? 'x'.repeat(5_000) : `reply ${i}`, `h${i}`);
    }
    const tools = captureTools(registerSearchTools, { db, cfg: {}, searchContext: {} });
    call = async (name, args) => JSON.parse((await tools[name](args)).content[0].text);
  });
  afterEach(() => db.close());

  it('m9k_full answers in the order asked, each id once', async () => {
    const r = (await call('m9k_full', {
      chunkIds: ['s:5', 's:1', 's:5', 'nope', 's:0'],
    })) as Array<{
      id: string;
    }>;
    expect(r.map((x) => x.id)).toEqual(['s:5', 's:1', 's:0']);
  });

  it('with a budget, m9k_full returns whole chunks, the missing ids and the rest to fetch next', async () => {
    const r = (await call('m9k_full', {
      chunkIds: ['s:1', 's:3', 's:2', 'gone'],
      budgetChars: 1_000,
    })) as {
      chunks: Array<{ id: string; assistant_content: string }>;
      missing: string[];
      next: string[];
    };
    expect(r.chunks.map((x) => x.id)).toEqual(['s:1']);
    expect(r.missing).toEqual(['gone']);
    expect(r.next).toEqual(['s:3', 's:2']);

    const again = (await call('m9k_full', { chunkIds: r.next, budgetChars: 1_000 })) as typeof r;
    expect(again.chunks[0].assistant_content).toHaveLength(5_000); // over budget alone: whole, never cut
    expect(again.next).toEqual(['s:2']);
  });

  it('with a budget, m9k_context keeps the target and its nearest neighbours', async () => {
    const r = (await call('m9k_context', { chunkId: 's:1', window: 3, budgetChars: 1_000 })) as {
      chunks: Array<{ id: string }>;
      omitted: string[];
    };
    expect(r.chunks.map((x) => x.id)).toEqual(['s:0', 's:1', 's:2']);
    expect(r.omitted).toEqual(['s:3', 's:4']);

    const plain = (await call('m9k_context', { chunkId: 's:1', window: 3 })) as {
      chunks: unknown[];
      omitted?: unknown;
    };
    expect(plain.chunks).toHaveLength(5);
    expect(plain.omitted).toBeUndefined(); // without a budget the reply is as before
  });
});
