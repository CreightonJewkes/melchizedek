import { describe, it, expect, afterEach } from 'vitest';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { initLogger, closeLogger, logger } from '../src/logger.js';

const waitFor = async (check: () => boolean) => {
  for (let i = 0; i < 50 && !check(); i++) await new Promise((r) => setTimeout(r, 20));
};

describe('logging (C4)', () => {
  const dirs: string[] = [];
  const tmp = () => {
    const d = fs.mkdtempSync(path.join(os.tmpdir(), 'm9k-log-'));
    dirs.push(d);
    return d;
  };
  afterEach(() => {
    closeLogger();
    initLogger({ file: false }); // back to the default level, no file
    for (const d of dirs.splice(0)) fs.rmSync(d, { recursive: true, force: true });
  });

  it('the file gets what the level allows, and nothing below it', async () => {
    const dir = tmp();
    initLogger({ level: 'info', logDir: dir });
    logger.debug('t', 'hidden debug line');
    logger.info('t', 'backfill complete');
    logger.warn('t', 'a warning');
    const file = path.join(dir, 'melchizedek.log');
    await waitFor(() => fs.existsSync(file) && fs.readFileSync(file, 'utf8').includes('a warning'));
    const text = fs.readFileSync(file, 'utf8');
    expect(text).toContain('[INFO] [t] backfill complete');
    expect(text).not.toContain('hidden debug line');
  });

  it('file: false logs to stderr only (embed workers), creating no file', async () => {
    const dir = tmp();
    initLogger({ level: 'debug', logDir: dir, file: false });
    logger.error('t', 'to stderr');
    await new Promise((r) => setTimeout(r, 50));
    expect(fs.readdirSync(dir)).toEqual([]);
  });

  it('the embed worker no longer appends to the log file around the logger', () => {
    const src = fs.readFileSync(path.join(import.meta.dirname, '../src/embed-worker.ts'), 'utf8');
    expect(src).not.toMatch(/appendFileSync|melchizedek\.log/);
    expect(src).toMatch(/file: false/);
  });
});
