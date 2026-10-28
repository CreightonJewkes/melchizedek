/**
 * SocketTransport — MCP Transport implementation over a net.Socket.
 * Uses newline-delimited JSON framing (same as StdioServerTransport).
 */

import type net from 'net';
import type {
  Transport,
  TransportSendOptions,
} from '@modelcontextprotocol/sdk/shared/transport.js';
import type { JSONRPCMessage } from '@modelcontextprotocol/sdk/types.js';
import { JSONRPCMessageSchema } from '@modelcontextprotocol/sdk/types.js';

const NEWLINE = 0x0a;

/** A frame this large without a newline is refused: the peer is broken or hostile. */
export const MAX_FRAME_BYTES = 64 * 1024 * 1024;

/**
 * Split the first newline-terminated line off a buffer. Framing works on bytes: UTF-8 never
 * uses 0x0A inside a multibyte character, so a whole line always decodes cleanly, however
 * the stream was chunked. Returns null until a newline arrives.
 */
export function splitFirstLine(buf: Buffer): { line: string; rest: Buffer } | null {
  const idx = buf.indexOf(NEWLINE);
  if (idx === -1) return null;
  return { line: buf.subarray(0, idx).toString('utf8'), rest: buf.subarray(idx + 1) };
}

export class SocketTransport implements Transport {
  /** Bytes of the current, unterminated frame. */
  private pending: Buffer[] = [];
  private pendingBytes = 0;
  private started = false;
  private readonly maxFrameBytes: number;
  sessionId?: string;

  onmessage?: (message: JSONRPCMessage) => void;
  onclose?: () => void;
  onerror?: (error: Error) => void;

  constructor(
    private socket: net.Socket,
    options?: { maxFrameBytes?: number },
  ) {
    this.maxFrameBytes = options?.maxFrameBytes ?? MAX_FRAME_BYTES;
  }

  async start(): Promise<void> {
    if (this.started) {
      throw new Error('SocketTransport already started');
    }
    this.started = true;

    this.socket.on('data', (chunk: Buffer) => this.receive(chunk));

    this.socket.on('close', () => {
      this.onclose?.();
    });

    this.socket.on('error', (err: Error) => {
      this.onerror?.(err);
    });
  }

  private receive(chunk: Buffer): void {
    let start = 0;
    let idx: number;
    while ((idx = chunk.indexOf(NEWLINE, start)) !== -1) {
      const tail = chunk.subarray(start, idx);
      const frame = this.pending.length > 0 ? Buffer.concat([...this.pending, tail]) : tail;
      this.pending = [];
      this.pendingBytes = 0;
      this.handleLine(frame.toString('utf8').replace(/\r$/, ''));
      start = idx + 1;
    }
    if (start < chunk.length) {
      const rest = chunk.subarray(start);
      this.pendingBytes += rest.length;
      if (this.pendingBytes > this.maxFrameBytes) {
        this.pending = [];
        this.pendingBytes = 0;
        this.onerror?.(new Error(`Frame over ${this.maxFrameBytes} bytes without a newline`));
        this.socket.destroy();
        return;
      }
      this.pending.push(rest);
    }
  }

  private handleLine(line: string): void {
    if (line.length === 0) return;
    try {
      const message = JSONRPCMessageSchema.parse(JSON.parse(line));
      this.onmessage?.(message);
    } catch (err) {
      this.onerror?.(err instanceof Error ? err : new Error(String(err)));
    }
  }

  /** Resolves once the data is handed to the socket; rejects if it closes or fails first. */
  async send(message: JSONRPCMessage, _options?: TransportSendOptions): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      if (this.socket.destroyed) {
        reject(new Error('Socket is destroyed'));
        return;
      }
      const data = JSON.stringify(message) + '\n';
      if (this.socket.write(data)) {
        resolve();
        return;
      }
      const settle = (err?: Error) => {
        this.socket.off('drain', onDrain);
        this.socket.off('close', onClose);
        this.socket.off('error', onError);
        if (err) reject(err);
        else resolve();
      };
      const onDrain = () => settle();
      const onClose = () => settle(new Error('Socket closed before the message was sent'));
      const onError = (err: Error) => settle(err);
      this.socket.once('drain', onDrain);
      this.socket.once('close', onClose);
      this.socket.once('error', onError);
    });
  }

  async close(): Promise<void> {
    this.socket.end();
  }
}
