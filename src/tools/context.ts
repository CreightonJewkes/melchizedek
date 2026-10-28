/**
 * Shared context passed to all tool registration functions.
 */

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
