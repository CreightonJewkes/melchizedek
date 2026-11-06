import { describe, it, expect } from 'vitest';
import { spawnSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';

/**
 * A second daemon started while one already runs (launchd's copy after a proxy started one
 * first) must exit 0, so a supervisor that restarts only on failure leaves it be instead of
 * relaunching it every throttle interval. Runs the built daemon, so `npm run build` comes first.
 */
const DIST = path.join(import.meta.dirname, '..', 'dist', 'daemon.js');

describe.skipIf(process.platform === 'win32' || !fs.existsSync(DIST))(
  'daemon already running',
  () => {
    it('exits 0 and leaves the live daemon’s PID file and socket alone', () => {
      const home = fs.mkdtempSync(path.join(os.tmpdir(), 'm9k-dup-'));
      const dir = path.join(home, '.melchizedek');
      fs.mkdirSync(dir);
      const pidFile = path.join(dir, 'daemon.pid');
      fs.writeFileSync(pidFile, String(process.pid)); // a live process stands in for the running daemon

      const r = spawnSync(process.execPath, [DIST], {
        env: { ...process.env, HOME: home, USERPROFILE: home },
        encoding: 'utf8',
        timeout: 20_000,
      });

      expect(r.status).toBe(0);
      expect(r.stderr).toContain(`Daemon already running (pid ${process.pid})`);
      expect(fs.readFileSync(pidFile, 'utf8')).toBe(String(process.pid));
      expect(fs.existsSync(path.join(dir, 'daemon.sock'))).toBe(false);
    });
  },
);
