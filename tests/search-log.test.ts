import { describe, it, expect, afterEach } from 'vitest';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { z } from 'zod';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { logToolCalls, recordStage } from '../src/search-log.js';
import { createServer } from '../src/server.js';
import { closeDatabase, openMemoryDatabase } from '../src/db.js';
import { indexConvSession } from '../src/indexer.js';

const FIXTURE = path.join(import.meta.dirname, 'fixtures', 'normal_session.jsonl');
const lines = (f: string) =>
  fs.existsSync(f)
    ? fs
        .readFileSync(f, 'utf8')
        .trim()
        .split('\n')
        .filter(Boolean)
        .map((l) => JSON.parse(l))
    : [];

async function connect(server: McpServer): Promise<Client> {
  const client = new Client({ name: 'test-client', version: '1.0.0' });
  const [ct, st] = InMemoryTransport.createLinkedPair();
  await Promise.all([server.connect(st), client.connect(ct)]);
  return client;
}

describe('search log', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'm9k-searchlog-'));

  it('logs arguments, duration, result count, first IDs and stage timings', async () => {
    const file = path.join(tmp, 'a.jsonl');
    const server = new McpServer({ name: 't', version: '0' });
    logToolCalls(server, { file, session: 'sess-1', project: '/p' });
    server.registerTool('find', { inputSchema: { query: z.string() } }, async () => {
      recordStage('bm25', 12.4);
      return {
        content: [
          {
            type: 'text' as const,
            text: JSON.stringify([{ chunkId: 's1:0' }, { chunkId: 's2:3' }]),
          },
        ],
      };
    });
    server.registerTool('info', {}, async () => ({
      content: [{ type: 'text' as const, text: '{"ok":true}' }],
    }));
    const client = await connect(server);
    await client.callTool({ name: 'find', arguments: { query: 'fts5 hyphen' } });
    await client.callTool({ name: 'info', arguments: {} });

    const [find, info] = lines(file);
    expect(find).toMatchObject({
      tool: 'find',
      session: 'sess-1',
      project: '/p',
      args: { query: 'fts5 hyphen' },
    });
    expect(find).toMatchObject({ error: false, n: 2, ids: ['s1:0', 's2:3'], stages: { bm25: 12 } });
    expect(typeof find.ms).toBe('number');
    expect(info).toMatchObject({ tool: 'info', args: {}, n: null, ids: [] });
    expect(info.stages).toBeUndefined();
  });

  it('records a failing call and still fails it', async () => {
    const file = path.join(tmp, 'b.jsonl');
    const server = new McpServer({ name: 't', version: '0' });
    logToolCalls(server, { file });
    server.registerTool('boom', { inputSchema: { query: z.string() } }, async () => {
      throw new Error('nope');
    });
    const client = await connect(server);
    const r = await client.callTool({ name: 'boom', arguments: { query: 'x' } });
    expect(r.isError).toBe(true);
    expect(lines(file)[0]).toMatchObject({ tool: 'boom', error: true });
  });

  it('rotates the file once past its size cap', async () => {
    const file = path.join(tmp, 'c.jsonl');
    const server = new McpServer({ name: 't', version: '0' });
    logToolCalls(server, { file, maxBytes: 10 });
    server.registerTool('t', { inputSchema: { query: z.string() } }, async () => ({
      content: [{ type: 'text' as const, text: '[]' }],
    }));
    const client = await connect(server);
    for (const q of ['a', 'b', 'c']) await client.callTool({ name: 't', arguments: { query: q } });
    expect(lines(file).map((e) => e.args.query)).toEqual(['c']);
    expect(lines(`${file}.1`).map((e) => e.args.query)).toEqual(['b']);
  });

  describe('through createServer', () => {
    const logFile = path.join(os.homedir(), '.melchizedek', 'logs', 'search.jsonl');
    let db: ReturnType<typeof openMemoryDatabase>['db'];
    afterEach(() => closeDatabase(db));

    async function searchWith(overrides: { searchLog?: boolean }) {
      db = openMemoryDatabase().db;
      indexConvSession(
        db,
        '550e8400-e29b-41d4-a716-446655440000',
        fs.readFileSync(FIXTURE, 'utf8'),
        '/Users/test/my-project',
        '/path/to/session.jsonl',
      );
      const client = await connect(createServer(overrides, db).server);
      await client.callTool({ name: 'm9k_search', arguments: { query: 'test' } });
    }

    it('writes nothing by default', async () => {
      await searchWith({});
      expect(fs.existsSync(logFile)).toBe(false);
    });

    it('logs m9k_search with its stages when searchLog is on', async () => {
      await searchWith({ searchLog: true });
      const last = lines(logFile).at(-1);
      expect(last).toMatchObject({ tool: 'm9k_search', args: { query: 'test' }, error: false });
      expect(last.stages).toHaveProperty('bm25');
    });
  });
});
