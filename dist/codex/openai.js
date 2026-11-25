import { createRequire } from 'node:module';
import { s, parseJson } from '../core/schema.js';
import { MaintainerError } from '../core/errors.js';
import { redact } from '../core/security.js';
import { silentLogger } from '../core/logger.js';
import { SYSTEM_POLICY } from './provider.js';
export function createOfficialClient(apiKey, baseURL = process.env['CM_OPENAI_BASE_URL'] || 'https://api.openai.com/v1') {
    let imported;
    try {
        imported = createRequire(import.meta.url)('openai');
    }
    catch {
        throw new MaintainerError('SDK_MISSING', 'The official openai package is not installed. Run npm install in the Codex Maintainer installation directory.');
    }
    const constructor = typeof imported === 'function' ? imported : (typeof imported === 'object' && imported !== null && 'default' in imported ? imported.default : undefined);
    if (typeof constructor !== 'function')
        throw new MaintainerError('SDK_MISSING', 'The installed openai package does not expose its documented default constructor.');
    const OpenAI = constructor;
    return new OpenAI({ apiKey, baseURL, maxRetries: 2, timeout: 120_000, logLevel: 'off' });
}
// Some OpenAI-compatible gateways accept json_schema but still emit extra keys.
// Drop unknown keys recursively before the strict local schema check so required
// fields and bounds remain enforced without discarding an otherwise valid report.
function conform(value, schema) {
    if (Array.isArray(value) && schema['type'] === 'array') {
        const item = schema['items'];
        return item && typeof item === 'object' ? value.map((entry) => conform(entry, item)) : value;
    }
    if (value !== null && typeof value === 'object' && !Array.isArray(value) && schema['type'] === 'object') {
        const properties = schema['properties'];
        if (!properties || typeof properties !== 'object')
            return value;
        const result = {};
        for (const [key, child] of Object.entries(properties)) {
            if (Object.hasOwn(value, key) && child && typeof child === 'object')
                result[key] = conform(value[key], child);
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
export class OpenAIProvider {
    config;
    secrets;
    logger;
    client;
    constructor(config, apiKey, secrets = [], logger = silentLogger, client) {
        this.config = config;
        this.secrets = secrets;
        this.logger = logger;
        if (!apiKey.trim())
            throw new MaintainerError('CONFIG', 'OPENAI_API_KEY is required.');
        this.client = client ?? createOfficialClient(apiKey);
    }
    async complete(task, schema, evidence) {
        const input = redact(JSON.stringify({ task, untrustedEvidence: evidence }), this.secrets);
        if (input.length > 500_000)
            throw new MaintainerError('LIMIT', 'The assembled prompt exceeds its character budget. Lower the configured limits.');
        const effort = process.env['CM_REASONING_EFFORT'] || 'high';
        if (!['minimal', 'low', 'medium', 'high', 'xhigh'].includes(effort))
            throw new MaintainerError('CONFIG', 'Invalid reasoning effort.');
        this.logger.info(`Requesting structured ${task} analysis with ${this.config.model}.`);
        const required = schema.json['required'];
        const hint = Array.isArray(required) ? ` Return only the requested JSON object with exactly these top-level fields: ${required.join(', ')}.` : '';
        try {
            const raw = await this.client.responses.create({
                model: this.config.model, store: false, reasoning: { effort },
                input: [{ role: 'system', content: SYSTEM_POLICY + hint }, { role: 'user', content: input }],
                text: { format: { type: 'json_schema', name: `codex_maintainer_${task}`, strict: true, schema: schema.json } },
                max_output_tokens: this.config.maxOutputTokens,
            }, { signal: AbortSignal.timeout(180_000) });
            const response = responseSchema.parse(raw);
            const messages = response.output.filter((item) => item.type === 'message');
            if (messages.some((message) => message.content.some((part) => part.type === 'refusal')))
                throw new MaintainerError('REFUSAL', 'The model declined this request. No comment was published.');
            if (response.status !== 'completed')
                throw new MaintainerError('INCOMPLETE', 'The model response was incomplete or failed. Increase maxOutputTokens, narrow the input, or retry. No partial report was published.');
            if (response.output.some((item) => !['reasoning', 'message'].includes(item.type)))
                throw new MaintainerError('MODEL_OUTPUT', 'Unexpected model tool output was rejected.');
            const finals = messages.filter((item) => item.phase === 'final_answer');
            const selected = finals.length ? finals : messages;
            const extracted = selected.flatMap((item) => item.content.filter((part) => part.type === 'output_text').map((part) => part.text)).join('');
            const text = redact(extracted || response.output_text, this.secrets);
            let parsed;
            try {
                parsed = JSON.parse(text);
            }
            catch {
                throw new MaintainerError('MODEL_OUTPUT', 'The model did not return valid JSON. No comment was published.');
            }
            let value;
            try {
                value = parseJson(JSON.stringify(conform(parsed, schema.json)), schema, true);
            }
            catch {
                throw new MaintainerError('MODEL_OUTPUT', `The model JSON did not match the required schema. Received: ${text.slice(0, 1200)}`);
            }
            return { value, provider: 'openai', model: this.config.model, usage: response.usage ? { inputTokens: response.usage.input_tokens, outputTokens: response.usage.output_tokens } : null, requestId: response._request_id };
        }
        catch (error) {
            if (error instanceof MaintainerError) {
                if (error.code === 'INVALID_DATA')
                    throw new MaintainerError('MODEL_OUTPUT', 'OpenAI returned an unexpected response shape. No comment was published.');
                throw error;
            }
            const status = typeof error === 'object' && error !== null && 'status' in error && typeof error.status === 'number' ? error.status : undefined;
            if (status === 429)
                throw new MaintainerError('RATE_LIMIT', 'OpenAI rate limit or quota was exceeded after bounded SDK retries. Check API billing/limits and retry later.', status);
            if (status === 401 || status === 403)
                throw new MaintainerError('AUTH', 'OpenAI rejected the key or access to the configured model.', status);
            if (status === 400 || status === 404)
                throw new MaintainerError('CONFIG', 'OpenAI rejected the configured model or request. Choose a model with Responses API and Structured Outputs support.', status);
            throw new MaintainerError('NETWORK', 'OpenAI request failed or timed out after bounded retries. No comment was published.', status);
        }
    }
}
//# sourceMappingURL=openai.js.map