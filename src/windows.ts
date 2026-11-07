/**
 * Split prompt/reply windows: an option of the text embedding recipe (config
 * embeddingTextWindows). With it on, a chunk's prompt and reply are embedded apart, so a
 * long reply no longer crowds the prompt out of one capped input, and a query matches
 * whichever half it is about.
 *
 * Storage keeps one vector per chunk per table pair: the first window lives in the
 * generation's own pair (conv_vec_text, conv_vec_map_text), the second in a parallel pair
 * (conv_vec_text_w2, conv_vec_map_text_w2). Deletion, the orphan sweep, compaction and
 * health enumerate every conv_vec_map* pair, so they cover the second window unchanged;
 * the embedded and pending counts read the first pair, one row per chunk.
 */

import type Database from 'better-sqlite3';
import type { Embedder } from './models.js';

/** The second-window pair beside a generation: _text -> _text_w2, _text_next -> _text_w2_next. */
export function secondWindowSuffix(suffix: string): string {
  return suffix.endsWith('_next') ? `${suffix.slice(0, -'_next'.length)}_w2_next` : `${suffix}_w2`;
}

/** The generation a table suffix belongs to (_text_w2_next -> _text), for migration checks. */
export function baseSuffix(suffix: string): string {
  return suffix.replace(/_next$/, '').replace(/_w2$/, '');
}

/** True when this generation holds second-window vectors. */
export function hasSecondWindow(db: Database.Database, suffix: string): boolean {
  return !!db
    .prepare("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?")
    .get(`conv_vec_map${secondWindowSuffix(suffix)}`);
}

/**
 * The texts embedded for one chunk: the exchange joined, as always, or with windows on the
 * prompt and the reply apart. A window with no text is left out, so a chunk with only one
 * side has one vector.
 */
export function chunkWindows(user: string, assistant: string, split: boolean): string[] {
  const joined = `${user} ${assistant}`;
  if (!split) return [joined];
  const windows = [user, assistant].filter((t) => t.trim().length > 0);
  return windows.length > 0 ? windows : [joined];
}

/** Embed chunks window by window in one batch; embedding2 is the reply window, if any. */
export async function embedChunkWindows(
  embedder: Embedder,
  chunks: Array<{ user: string; assistant: string }>,
): Promise<Array<{ embedding: Float32Array; embedding2?: Float32Array }>> {
  const split = embedder.splitWindows?.() ?? false;
  const windows = chunks.map((c) => chunkWindows(c.user, c.assistant, split));
  const vectors = await embedder.embedBatch(windows.flat());
  let i = 0;
  return windows.map((w) => {
    const out = { embedding: vectors[i], embedding2: w.length > 1 ? vectors[i + 1] : undefined };
    i += w.length;
    return out;
  });
}
