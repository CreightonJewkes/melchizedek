import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type Database from 'better-sqlite3';
import { openMemoryDatabase, setStat } from '../src/db.js';
import { getConfig } from '../src/config.js';
import { createServer, knownSessionId, registerMessage, sessionIdFromEnv } from '../src/server.js';
import { EmbedOrchestrator } from '../src/embed-orchestrator.js';
import { registerSearchTools } from '../src/tools/search.js';
import { registerSpecializedTools } from '../src/tools/specialized.js';
import type { SearchContext } from '../src/models.js';
import { captureTools } from './helpers/fixtures.js';

const searchContext = (): SearchContext => ({
  embedderText: null,
  embedderCode: null,
  reranker: null,
  vecTextEnabled: false,
  vecCodeEnabled: false,
  autoFuzzyThreshold: 3,
});

describe('session boost per connection; project boost from config (C1, D22)', () => {
  let db: Database.Database;
  const connect = (currentSession?: string, currentProject = '/p2') => {
    const ctx = {
      db,
      cfg: getConfig({ dbPath: ':memory:' }),
      searchContext: searchContext(),
      currentProject,
      currentSession,
    };
    return {
      ctx,
      tools: {
        ...captureTools(registerSearchTools, ctx),
        ...captureTools(registerSpecializedTools, ctx),
      },
    };
  };
  const first = async (
    tools: ReturnType<typeof connect>['tools'],
    name = 'm9k_search',
    args: Record<string, unknown> = { query: 'kafka consumer lag', limit: 5 },
  ) =>
    (JSON.parse((await tools[name](args)).content[0].text) as Array<{ sessionId: string }>)[0]
      .sessionId;

  beforeEach(() => {
    db = openMemoryDatabase().db;
    db.prepare(
      `INSERT INTO conv_sessions (id, project, jsonl_path, started_at) VALUES
       ('sess-a', '/p1', '/t/a', '2026-01-01'), ('sess-b', '/p2', '/t/b', '2026-01-02')`,
    ).run();
    const ins = db.prepare(
      `INSERT INTO conv_chunks (id, session_id, idx, user_content, assistant_content, hash, timestamp)
       VALUES (?, ?, 0, ?, 'raise the session timeout and check the partition assignment', ?, ?)`,
    );
    // Two equally good matches, one per session (different text, so not collapsed as copies).
    ins.run(
      'sess-a:0',
      'sess-a',
      'kafka consumer lag on the orders topic',
      'ha',
      '2026-01-01T00:00:00Z',
    );
    ins.run(
      'sess-b:0',
      'sess-b',
      'kafka consumer lag on the billing topic',
      'hb',
      '2026-01-02T00:00:00Z',
    );
  });
  afterEach(() => db.close());

  it("each connection is boosted by its own session, whatever another session's Stop hook does", async () => {
    const a = connect('sess-a');
    const b = connect('sess-b');
    expect(await first(a.tools)).toBe('sess-a');
    setStat(db, 'current_session_id', ''); // another session's Stop hook (1.0.2 read this stat)
    expect(await first(b.tools)).toBe('sess-b');
    expect(await first(a.tools)).toBe('sess-a');
  });

  it('m9k_errors and m9k_similar_work stay boost-neutral', async () => {
    const a = connect('sess-a', '/p1');
    const b = connect('sess-b', '/p2');
    a.ctx.cfg.projectBoost = b.ctx.cfg.projectBoost = 1.5;
    for (const [name, args] of [
      ['m9k_errors', { errorMessage: 'kafka consumer lag', limit: 5 }],
      ['m9k_similar_work', { description: 'kafka consumer lag', limit: 5 }],
    ] as const) {
      expect(await first(a.tools, name, args)).toBe(await first(b.tools, name, args));
    }
  });

  it('the project boost is off by default and comes from config', async () => {
    const c = connect(undefined, '/p1');
    expect(c.ctx.cfg.projectBoost).toBe(1);
    const neutral = await first(c.tools);
    const other = neutral === 'sess-a' ? '/p2' : '/p1';
    const d = connect(undefined, other);
    expect(await first(d.tools)).toBe(neutral); // 1.0: the caller's project changes nothing
    d.ctx.cfg.projectBoost = 1.5; // the ranking of earlier releases
    expect(await first(d.tools)).not.toBe(neutral);
  });

  it('every connection shares one search context', () => {
    const shared = searchContext();
    const one = createServer({ dbPath: ':memory:' }, db, { searchContext: shared });
    const two = createServer({ dbPath: ':memory:' }, db, { searchContext: shared });
    expect(one.searchContext).toBe(shared);
    expect(two.searchContext).toBe(shared);
  });
});

describe('the proxy registers the Claude Code session (C1)', () => {
  it('reads CLAUDE_CODE_SESSION_ID, then SESSION_ID; "unknown" counts as absent', () => {
    expect(sessionIdFromEnv({ CLAUDE_CODE_SESSION_ID: 'uuid-1', SESSION_ID: 'x' })).toBe('uuid-1');
    expect(sessionIdFromEnv({ SESSION_ID: 'uuid-2' })).toBe('uuid-2');
    expect(sessionIdFromEnv({ CLAUDE_CODE_SESSION_ID: 'unknown' })).toBeUndefined();
    expect(knownSessionId('')).toBeUndefined();
    expect(JSON.parse(registerMessage('/proj', { CLAUDE_CODE_SESSION_ID: 'uuid-1' }))).toEqual({
      type: 'register',
      project: '/proj',
      sessionId: 'uuid-1',
    });
  });
});

describe('embed status reaches every connection (C1)', () => {
  it('status listeners are a set, each with its own unsubscribe', () => {
    const db = openMemoryDatabase().db;
    const orch = new EmbedOrchestrator('/tmp/x.db', db, { workerPath: '/fake' });
    const seen: string[] = [];
    const offA = orch.onStatusChange(() => seen.push('a'));
    orch.onStatusChange(() => seen.push('b'));
    orch.abort(); // resets status, which notifies
    expect(seen).toEqual(['a', 'b']);
    offA();
    orch.abort();
    expect(seen).toEqual(['a', 'b', 'b']);
    db.close();
  });
});
