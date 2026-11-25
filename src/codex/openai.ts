import { createRequire } from 'node:module';
import { s, parseJson, type Schema } from '../core/schema.js';
import { MaintainerError } from '../core/errors.js';
import { redact } from '../core/security.js';
import { type Config } from '../core/config.js';
import { type Logger, silentLogger } from '../core/logger.js';
import { type AIProvider, type AIResult, type Task, SYSTEM_POLICY } from './provider.js';

/** Structural boundary keeps SDK-specific types out of the maintenance domain. */
export interface ResponsesClient {
  responses: { create(body: Record<string, unknown>, options?: { signal: AbortSignal }): Promise<unknown> };
}
export interface SdkOptions { apiKey: string; baseURL: string; maxRetries: number; timeout: number; logLevel: 'off' }
export function createOfficialClient(apiKey: string, baseURL: string = process.env['CM_OPENAI_BASE_URL'] || 'https://api.openai.com/v1'): ResponsesClient {
  let imported: unknown;
  try { imported = createRequire(import.meta.url)('openai') as unknown; }
  catch { throw new MaintainerError('SDK_MISSING', 'The official openai package is not installed. Run npm install in the Codex Maintainer installation directory.'); }
  const constructor = typeof imported === 'function' ? imported : (typeof imported === 'object' && imported !== null && 'default' in imported ? imported.default : undefined);
  if (typeof constructor !== 'function') throw new MaintainerError('SDK_MISSING', 'The installed openai package does not expose its documented default constructor.');
  const OpenAI = constructor as new (options: SdkOptions) => ResponsesClient;
  return new OpenAI({ apiKey, baseURL, maxRetries: 2, timeout: 120_000, logLevel: 'off' });
}

function defaultFor(schema: Record<string, unknown>): unknown {
  const anyOf = schema['anyOf'];
  if (Array.isArray(anyOf) && anyOf.length) return defaultFor(anyOf[0] as Record<string, unknown>);
  // Never invent a category, severity, confidence or changed-line citation.
  if (Array.isArray(schema['enum'])) return undefined;
  switch (schema['type']) {
    case 'string': {
      const min = typeof schema['minLength'] === 'number' ? schema['minLength'] as number : 0;
      const max = typeof schema['maxLength'] === 'number' ? schema['maxLength'] as number : 20_000;
      return 'Not specified by the model.'.padEnd(min, '.').slice(0, Math.max(min, max));
    }
    case 'integer':
    case 'number': return typeof schema['minimum'] === 'number' ? schema['minimum'] : 0;
    case 'boolean': return false;
    case 'array': return [];
    case 'object': return {};
    default: return null;
  }
}

// Some OpenAI-compatible gateways accept json_schema but still emit extra keys.
// Normalize known field aliases and drop unknown keys before the strict local
// schema check so required fields and bounds remain enforced.
function conform(value: unknown, schema: Record<string, unknown>): unknown {
  if (Array.isArray(value) && schema['type'] === 'array') {
    const item = schema['items'];
    if (!item || typeof item !== 'object') return value;
    return value.map((entry) => conform(entry, item as Record<string, unknown>)).filter((entry) => entry !== undefined);
  }
  if (value !== null && typeof value === 'object' && !Array.isArray(value) && schema['type'] === 'object') {
    const properties = schema['properties'];
    if (!properties || typeof properties !== 'object') return value;
    const source = value as Record<string, unknown>;
    const result: Record<string, unknown> = {};
    for (const [key, child] of Object.entries(properties)) {
      if (!child || typeof child !== 'object') continue;
      const field = child as Record<string, unknown>;
      let raw = Object.hasOwn(source, key) ? source[key] : undefined;
      // A finding without a changed-line citation is not a finding: drop the
      // whole object rather than filling in a fabricated location or category.
      const locationKeys = ['path', 'side', 'line', 'quote', 'category', 'severity', 'confidence'];
      if (raw === undefined && locationKeys.includes(key) && Array.isArray(schema['required']) && (schema['required'] as unknown[]).includes(key)) return undefined;
      if (raw === undefined && key === 'explanation') raw = [source['trigger'], source['impact'], source['reason'], source['issue']].filter((x) => typeof x === 'string').join('\n');
      if (raw === undefined && key === 'recommendation') raw = [source['remedy'], source['fix']].filter((x) => typeof x === 'string').join('\n');
      if (raw === undefined) raw = defaultFor(field);
      if (raw === undefined) return undefined;
      result[key] = conform(raw, field);
    }
    return result;
  }
  return value;
}

const contentSchema = s.object({ type: s.string(100), text: s.optional(s.string(500_000), ''), refusal: s.optional(s.string(20_000), '') }, false);
const responseSchema = s.object({
  status: s.string(100), output_text: s.optional(s.string(500_000), ''),
  output: s.array(s.object({ type: s.string(100), phase: s.optional(s.nullable(s.string(100)), null), content: s.optional(s.array(contentSchema, 100), []) }, false), 200),
  usage: s.optional(s.nullable(s.object({ input_tokens: s.number(), output_tokens: s.number() }, false)), null),
  _request_id: s.optional(s.nullable(s.string(200)), null),
}, false);

export class OpenAIProvider implements AIProvider {
  private readonly client: ResponsesClient;
  constructor(private readonly config: Config, apiKey: string, private readonly secrets: readonly string[] = [], private readonly logger: Logger = silentLogger, client?: ResponsesClient) {
    if (!apiKey.trim()) throw new MaintainerError('CONFIG', 'OPENAI_API_KEY is required.');
    this.client = client ?? createOfficialClient(apiKey);
  }
  async complete<T>(task: Task, schema: Schema<T>, evidence: unknown): Promise<AIResult<T>> {
    const verbose = process.env['CM_VERBOSE'] === 'true';
    const input = redact(JSON.stringify({ task, untrustedEvidence: evidence }), this.secrets);
    if (input.length > 500_000) throw new MaintainerError('LIMIT', 'The assembled prompt exceeds its character budget. Lower the configured limits.');
    const effort = process.env['CM_REASONING_EFFORT'] || 'high';
    if (!['minimal', 'low', 'medium', 'high', 'xhigh'].includes(effort)) throw new MaintainerError('CONFIG', 'Invalid reasoning effort.');
    const rawTimeout = process.env['CM_REQUEST_TIMEOUT_MS'];
    const requestTimeout = rawTimeout ? Number(rawTimeout) : 180_000;
    if (!Number.isInteger(requestTimeout) || requestTimeout < 1_000 || requestTimeout > 900_000) throw new MaintainerError('CONFIG', 'request-timeout-ms must be an integer between 1000 and 900000.');
    this.logger.info(`Requesting structured ${task} analysis with ${this.config.model} (effort=${effort}, maxOutputTokens=${this.config.maxOutputTokens}, baseURL=${process.env['CM_OPENAI_BASE_URL'] || 'https://api.openai.com/v1'}).`);
    if (verbose) this.logger.info(`[verbose] prompt chars=${input.length}; required fields=${JSON.stringify((schema.json as Record<string, unknown>)['required'] ?? [])}`);
    const required = (schema.json as Record<string, unknown>)['required'];
    const hint = Array.isArray(required) ? ` Return only the requested JSON object with exactly these top-level fields: ${required.join(', ')}.` : '';
    try {
      const raw = await this.client.responses.create({
        model: this.config.model, store: false, reasoning: { effort },
        input: [{ role: 'system', content: SYSTEM_POLICY + hint }, { role: 'user', content: input }],
        text: { format: { type: 'json_schema', name: `codex_maintainer_${task}`, strict: true, schema: schema.json } },
        max_output_tokens: this.config.maxOutputTokens,
      }, { signal: AbortSignal.timeout(requestTimeout) });
      const response = responseSchema.parse(raw);
      if (verbose) this.logger.info(`[verbose] response status=${response.status}; item types=${JSON.stringify(response.output.map((item) => item.type))}; usage=${JSON.stringify(response.usage)}; requestId=${response._request_id ?? 'none'}`);
      const messages = response.output.filter((item) => item.type === 'message');
      if (messages.some((message) => message.content.some((part) => part.type === 'refusal'))) throw new MaintainerError('REFUSAL', 'The model declined this request. No comment was published.');
      if (response.status !== 'completed') throw new MaintainerError('INCOMPLETE', 'The model response was incomplete or failed. Increase maxOutputTokens, narrow the input, or retry. No partial report was published.');
      if (response.output.some((item) => !['reasoning', 'message'].includes(item.type))) throw new MaintainerError('MODEL_OUTPUT', 'Unexpected model tool output was rejected.');
      const finals = messages.filter((item) => item.phase === 'final_answer');
      const selected = finals.length ? finals : messages;
      const extracted = selected.flatMap((item) => item.content.filter((part) => part.type === 'output_text').map((part) => part.text)).join('');
      const text = redact(extracted || response.output_text, this.secrets);
      if (verbose) this.logger.info(`[verbose] model JSON: ${text.slice(0, 2000)}`);
      let parsed: unknown;
      try { parsed = JSON.parse(text) as unknown; } catch { throw new MaintainerError('MODEL_OUTPUT', 'The model did not return valid JSON. No comment was published.'); }
      let value: T;
      try { value = parseJson(JSON.stringify(conform(parsed, schema.json as Record<string, unknown>)), schema, true); }
      catch { throw new MaintainerError('MODEL_OUTPUT', `The model JSON did not match the required schema. Received: ${text.slice(0, 1200)}`); }
      return { value, provider: 'openai', model: this.config.model, usage: response.usage ? { inputTokens: response.usage.input_tokens, outputTokens: response.usage.output_tokens } : null, requestId: response._request_id };
    } catch (error) {
      if (error instanceof MaintainerError) {
        if (error.code === 'INVALID_DATA') throw new MaintainerError('MODEL_OUTPUT', 'OpenAI returned an unexpected response shape. No comment was published.');
        throw error;
      }
      const status = typeof error === 'object' && error !== null && 'status' in error && typeof error.status === 'number' ? error.status : undefined;
      if (status === 429) throw new MaintainerError('RATE_LIMIT', 'OpenAI rate limit or quota was exceeded after bounded SDK retries. Check API billing/limits and retry later.', status);
      if (status === 401 || status === 403) throw new MaintainerError('AUTH', 'OpenAI rejected the key or access to the configured model.', status);
      if (status === 400 || status === 404) throw new MaintainerError('CONFIG', 'OpenAI rejected the configured model or request. Choose a model with Responses API and Structured Outputs support.', status);
      throw new MaintainerError('NETWORK', 'OpenAI request failed or timed out after bounded retries. No comment was published.', status);
    }
  }
}
