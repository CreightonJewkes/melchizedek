/**
 * Date bounds for search and session filters.
 *
 * Chunk timestamps are stored as ISO-8601 UTC ('2026-03-01T09:00:00.000Z') and compared as
 * strings, so every bound is converted to exactly that form. A date or date-time without
 * an offset, and every relative expression, is read in local time: "since 2026-10-13"
 * means from local midnight. Relative expressions resolve against an injectable `now`,
 * so a replayed query resolves against the moment it was asked.
 */

export class DateInputError extends Error {}

export interface DateRange {
  since?: string;
  until?: string;
}

const ISO_RE =
  /^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.(\d+))?)?)?\s*(Z|[+-]\d{2}:?\d{2})?$/i;

/** Parse an absolute ISO-8601 date or date-time; null when it is not one or not a real date. */
export function parseAbsoluteDate(input: string): Date | null {
  const m = ISO_RE.exec(input.trim());
  if (!m) return null;
  const [y, mo, d, h, mi, s] = m.slice(1, 7).map((v) => parseInt(v ?? '0', 10));
  const ms = parseInt((m[7] ?? '0').slice(0, 3).padEnd(3, '0'), 10);
  const tz = m[8];
  if (mo < 1 || mo > 12 || d < 1 || d > 31 || h > 23 || mi > 59 || s > 59) return null;

  let date: Date;
  if (tz) {
    const offset = tz.toUpperCase() === 'Z' ? 'Z' : tz.replace(/^([+-]\d{2}):?(\d{2})$/, '$1:$2');
    const p2 = (n: number): string => String(n).padStart(2, '0');
    date = new Date(
      `${m[1]}-${m[2]}-${m[3]}T${p2(h)}:${p2(mi)}:${p2(s)}.${String(ms).padStart(3, '0')}${offset}`,
    );
  } else {
    date = new Date(y, mo - 1, d, h, mi, s, ms);
  }
  if (Number.isNaN(date.getTime())) return null;
  // Reject rollovers such as 2026-02-31 (local construction would silently move to March).
  if (!tz && (date.getFullYear() !== y || date.getMonth() !== mo - 1 || date.getDate() !== d)) {
    return null;
  }
  return date;
}

type Unit = 'minute' | 'hour' | 'day' | 'week' | 'month';

const UNITS: Record<string, Unit> = {
  minute: 'minute',
  min: 'minute',
  hour: 'hour',
  hr: 'hour',
  day: 'day',
  week: 'week',
  month: 'month',
};

/** "3", "a", "an", "a couple", "couple", "a few", "few" -> a count. */
const COUNT = String.raw`(\d+|an?|(?:a\s+)?couple|(?:a\s+)?few)`;
const UNIT = String.raw`(minute|min|hour|hr|day|week|month)s?`;
const AGO_RE = new RegExp(String.raw`^${COUNT}\s*(?:of\s+)?${UNIT}\s+ago$`);
const PAST_RE = new RegExp(String.raw`^(?:past|last)\s+${COUNT}\s*(?:of\s+)?${UNIT}$`);

function count(word: string): number {
  if (/^\d+$/.test(word)) return parseInt(word, 10);
  if (word.endsWith('couple')) return 2;
  if (word.endsWith('few')) return 3;
  return 1; // a / an
}

/** now minus n units: minutes and hours by elapsed time, days and longer by the local calendar. */
function minus(now: Date, n: number, unit: Unit): Date {
  const d = new Date(now.getTime());
  if (unit === 'minute') d.setTime(d.getTime() - n * 60_000);
  else if (unit === 'hour') d.setTime(d.getTime() - n * 3_600_000);
  else if (unit === 'day') d.setDate(d.getDate() - n);
  else if (unit === 'week') d.setDate(d.getDate() - 7 * n);
  else d.setMonth(d.getMonth() - n);
  return d;
}

const startOfDay = (d: Date, offsetDays = 0): Date =>
  new Date(d.getFullYear(), d.getMonth(), d.getDate() + offsetDays);

/**
 * One date bound: ISO-8601, or a relative point in local time resolved against `now`:
 * now, today, yesterday, "N units ago" (also "a", "a couple", "a few"), last week,
 * last month. Throws DateInputError for anything else.
 */
export function normalizeDateBound(
  input: string,
  field: 'since' | 'until',
  now: Date = new Date(),
): string {
  const date = parseAbsoluteDate(input);
  if (date) return date.toISOString();

  const t = input.trim().toLowerCase().replace(/\s+/g, ' ');
  if (t === 'now') return now.toISOString();
  if (t === 'today') return startOfDay(now).toISOString();
  if (t === 'yesterday') return startOfDay(now, -1).toISOString();
  if (t === 'last week') return minus(now, 1, 'week').toISOString();
  if (t === 'last month') return minus(now, 1, 'month').toISOString();
  const ago = AGO_RE.exec(t);
  if (ago) return minus(now, count(ago[1]), UNITS[ago[2]]).toISOString();
  throw new DateInputError(
    `Unrecognised ${field} date "${input}". Use ISO-8601 (2026-10-24, 2026-10-24T15:00:00Z) ` +
      'or a relative date: today, yesterday, "3 days ago", "2 hours ago", last week, last month.',
  );
}

/**
 * A named range in local time, until exclusive: today, yesterday, this/last week (weeks
 * start on Sunday), this/last month, "past N units", and "N units ago" (a window of one
 * unit either side). Throws DateInputError for anything else.
 */
export function resolveTimeframe(input: string, now: Date = new Date()): DateRange {
  const t = input.trim().toLowerCase().replace(/\s+/g, ' ');
  const iso = (d: Date): string => d.toISOString();
  if (t === 'today') return { since: iso(startOfDay(now)) };
  if (t === 'yesterday') return { since: iso(startOfDay(now, -1)), until: iso(startOfDay(now)) };
  if (t === 'this week') return { since: iso(startOfDay(now, -now.getDay())) };
  if (t === 'last week') {
    const thisWeek = startOfDay(now, -now.getDay());
    return { since: iso(startOfDay(thisWeek, -7)), until: iso(thisWeek) };
  }
  if (t === 'this month') return { since: iso(new Date(now.getFullYear(), now.getMonth(), 1)) };
  if (t === 'last month') {
    return {
      since: iso(new Date(now.getFullYear(), now.getMonth() - 1, 1)),
      until: iso(new Date(now.getFullYear(), now.getMonth(), 1)),
    };
  }
  const past = PAST_RE.exec(t);
  if (past) return { since: iso(minus(now, count(past[1]), UNITS[past[2]])) };
  const ago = AGO_RE.exec(t);
  if (ago) {
    const n = count(ago[1]);
    const unit = UNITS[ago[2]];
    return { since: iso(minus(now, n + 1, unit)), until: iso(minus(now, n - 1, unit)) };
  }
  throw new DateInputError(
    `Unsupported timeframe "${input}". Use today, yesterday, this week, last week, this month, ` +
      'last month, "past 2 hours", "past 3 days", or "3 hours ago" (one unit either side).',
  );
}

/**
 * Normalise since / until / timeframe for a tool call. A timeframe sets both bounds; an
 * explicit since or until overrides its side. Throws DateInputError for bad or reversed
 * input.
 */
export function resolveDateRange(
  input: { since?: string; until?: string; timeframe?: string },
  now: Date = new Date(),
): DateRange {
  const range: DateRange = input.timeframe ? resolveTimeframe(input.timeframe, now) : {};
  if (input.since) range.since = normalizeDateBound(input.since, 'since', now);
  if (input.until) range.until = normalizeDateBound(input.until, 'until', now);
  if (range.since && range.until && range.since >= range.until) {
    throw new DateInputError(`since (${range.since}) must be before until (${range.until}).`);
  }
  return range;
}

/** Standard MCP error result for a rejected date input. */
export function dateErrorResult(err: DateInputError): {
  content: Array<{ type: 'text'; text: string }>;
  isError: true;
} {
  return {
    content: [{ type: 'text' as const, text: JSON.stringify({ error: err.message }) }],
    isError: true,
  };
}
