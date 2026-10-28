import { describe, it, expect } from 'vitest';
import { EventEmitter } from 'events';
import type net from 'net';
import { SocketTransport, splitFirstLine } from '../src/socket-transport.js';
import type { JSONRPCMessage } from '@modelcontextprotocol/sdk/types.js';

/** A socket stand-in: the test drives 'data', write() reports backpressure as told. */
class FakeSocket extends EventEmitter {
  destroyed = false;
  writable = true;
  written: string[] = [];
  write(data: string): boolean {
    this.written.push(data);
    return this.writable;
  }
  destroy(): void {
    this.destroyed = true;
    this.emit('close');
  }
  end(): void {}
}

const message: JSONRPCMessage = {
  jsonrpc: '2.0',
  method: 'tools/call',
  id: 7,
  params: { name: 'm9k_save', arguments: { content: 'Le café — 日本語のテキスト 🎉 naïve' } },
};

describe('socket framing keeps multibyte text whole (C5)', () => {
  it('decodes the message identically whatever byte it is split at', async () => {
    const bytes = Buffer.from(JSON.stringify(message) + '\n', 'utf8');
    for (let cut = 1; cut < bytes.length; cut++) {
      const socket = new FakeSocket();
      const t = new SocketTransport(socket as unknown as net.Socket);
      const got: JSONRPCMessage[] = [];
      const errors: Error[] = [];
      t.onmessage = (m) => got.push(m);
      t.onerror = (e) => errors.push(e);
      await t.start();
      socket.emit('data', bytes.subarray(0, cut));
      socket.emit('data', bytes.subarray(cut));
      expect(errors, `cut at ${cut}`).toEqual([]);
      expect(got, `cut at ${cut}`).toEqual([message]);
    }
  });

  it('splitFirstLine (daemon registration) keeps the rest as raw bytes', () => {
    const reg = Buffer.from(
      JSON.stringify({ type: 'register', project: '/Users/zoë/项目' }) + '\n',
      'utf8',
    );
    const next = Buffer.from(JSON.stringify(message) + '\n', 'utf8');
    for (let cut = 1; cut < next.length; cut++) {
      const first = splitFirstLine(Buffer.concat([reg, next.subarray(0, cut)]))!;
      expect(JSON.parse(first.line).project).toBe('/Users/zoë/项目');
      expect(Buffer.concat([first.rest, next.subarray(cut)]).equals(next)).toBe(true);
    }
    expect(splitFirstLine(Buffer.from('no newline yet'))).toBeNull();
  });

  it('refuses a frame over the bound and closes the connection', async () => {
    const socket = new FakeSocket();
    const t = new SocketTransport(socket as unknown as net.Socket, { maxFrameBytes: 1_000 });
    const errors: Error[] = [];
    t.onerror = (e) => errors.push(e);
    await t.start();
    socket.emit('data', Buffer.alloc(600, 0x61));
    expect(socket.destroyed).toBe(false);
    socket.emit('data', Buffer.alloc(600, 0x61));
    expect(errors[0]?.message).toMatch(/without a newline/);
    expect(socket.destroyed).toBe(true);
  });

  it('a send waiting on backpressure settles on drain, close or error', async () => {
    const socket = new FakeSocket();
    const t = new SocketTransport(socket as unknown as net.Socket);
    await t.start();
    socket.writable = false;

    const drained = t.send(message);
    socket.emit('drain');
    await expect(drained).resolves.toBeUndefined();

    const closed = t.send(message);
    socket.emit('close');
    await expect(closed).rejects.toThrow(/closed/);

    const failed = t.send(message);
    socket.emit('error', new Error('EPIPE'));
    await expect(failed).rejects.toThrow('EPIPE');
    expect(socket.listenerCount('drain')).toBe(0); // no listener left behind
  });
});
