/**
 * Shared context passed to all tool registration functions.
 */

import { z } from 'zod';
import type { DatabaseType } from '../db.js';
import type { MelchizedekConfig, SearchContext } from '../models.js';
import type { EmbedOrchestrator } from '../embed-orchestrator.js';

export interface EmbeddingState {
  get(): boolean;
  set(v: boolean): void;
}

export interface ToolContext {
  db: DatabaseType;
  cfg: MelchizedekConfig;
  searchContext: SearchContext;
  currentProject?: string;
  /** The calling Claude Code session (per connection), for the session boost. */
  currentSession?: string;
  orchestrator: EmbedOrchestrator | null;
  embeddingState: EmbeddingState;
  version: string;
  mode: 'daemon' | 'local';
  /** Clock for relative dates ("yesterday"); injectable so replays resolve as asked. */
  clock?: () => Date;
}

/**
 * Conversations are the only indexed source. The parameter stays, with that one value, so
 * existing callers keep working; the values that were accepted and silently ignored
 * ('git', 'files') are now rejected.
 */
export const SOURCE_PARAM = z
  .enum(['conversations'])
  .optional()
  .describe("Accepted for compatibility: 'conversations' is the only source indexed.");
