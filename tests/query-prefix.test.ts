import { describe, it, expect } from 'vitest';
import { TransformersJsEmbedder } from '../src/embedder.js';
import { MODEL_REGISTRY } from '../src/constants.js';

/** An embedder whose pipeline records the texts it is given (no model is loaded). */
function recording(queryPrefix: string | undefined, maxInputChars = 2_000) {
  const seen: string[] = [];
  const e = new TransformersJsEmbedder({
    ...MODEL_REGISTRY['minilm-l12-v2'],
    queryPrefix,
    maxInputChars,
  });
  (e as unknown as { pipe: unknown }).pipe = async (texts: string[]) => {
    seen.push(...texts);
    return { tolist: () => texts.map(() => [1, 0]) };
  };
  return { e, seen };
}

describe('query prefix (A7)', () => {
  it('adds no second space after a prefix that ends in one', async () => {
    const { e, seen } = recording('query: ');
    await e.embedQuery('cors error');
    expect(seen).toEqual(['query: cors error']);
  });

  it('adds one separator after a prefix without trailing whitespace', async () => {
    const { e, seen } = recording(MODEL_REGISTRY['qwen3-embedding-0.6b'].queryPrefix);
    await e.embedQuery('cors error');
    expect(seen[0].endsWith('\nQuery: cors error')).toBe(true);

    const nl = recording('Instruct: find it\n');
    await nl.e.embedQuery('x');
    expect(nl.seen).toEqual(['Instruct: find it\nx']);
  });

  it('leaves the query unchanged when the model has no prefix', async () => {
    const { e, seen } = recording(undefined);
    await e.embedQuery('  spaced  ');
    expect(seen).toEqual(['  spaced  ']);
  });

  it('the input cap counts the prefix and cuts only the query text', async () => {
    const { e, seen } = recording('query: ', 20); // 7-character prefix
    await e.embedQuery('a'.repeat(13)); // exactly 20 with the prefix
    await e.embedQuery('b'.repeat(14)); // one over
    expect(seen[0]).toBe('query: ' + 'a'.repeat(13));
    expect(seen[1]).toBe('query: ' + 'b'.repeat(13));
  });

  it('every registry prefix joins with exactly one separator', async () => {
    for (const spec of Object.values(MODEL_REGISTRY)) {
      if (!spec.queryPrefix) continue;
      const { e, seen } = recording(spec.queryPrefix);
      await e.embedQuery('q');
      expect(seen[0], spec.key).not.toMatch(/\s{2}q$/);
      expect(seen[0], spec.key).toMatch(/\sq$/);
    }
  });
});
