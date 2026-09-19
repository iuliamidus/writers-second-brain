// Vercel Edge Function — streams Anthropic responses to the client so the
// chat feels alive. Kept small and stateless; adds the API key and enforces
// the password-hash header, then forwards.

export const config = { runtime: 'edge' };

const ANTHROPIC_URL = 'https://api.anthropic.com/v1/messages';
const ANTHROPIC_VERSION = '2023-06-01';
const DEFAULT_MODEL = 'claude-haiku-4-5-20251001';
const MAX_TOKENS = 2048;

const SYSTEM_PROMPT = `You are a research companion for a writer with four blogs and 706 posts spanning 2011 to today, mixed Romanian and English. Your job is to help the writer see the shape of the archive: recurring themes, cross-blog echoes, forgotten posts.

Use the tools to look things up rather than guessing. The archive is bilingual — respond in the language the user writes to you in. If the user writes in Romanian, respond in Romanian; if in English, respond in English.

Post ids are namespaced as {blog_slug}:{source_id}. Get them from search_posts before calling similar_posts or get_post. Titles alone are not ids.

When you cite posts, include the title, date, and blog. When appropriate, format post titles as clickable links using the url field returned by the tools.

Prefer concise, structural answers over long summaries. The user is a writer who values good prose.`;

const TOOLS = [
  {
    name: 'search_posts',
    description: 'Diacritic-insensitive keyword search across every blog. Returns id, title, url, blog, date, lang, and a highlighted snippet.',
    input_schema: {
      type: 'object',
      properties: {
        query: { type: 'string' },
        limit: { type: 'integer', default: 15, minimum: 1, maximum: 30 },
      },
      required: ['query'],
    },
  },
  {
    name: 'get_post',
    description: 'Full detail for a single post: title, body preview, url, tags, mentioned entities. Use the id from search_posts.',
    input_schema: {
      type: 'object',
      properties: { post_id: { type: 'string' } },
      required: ['post_id'],
    },
  },
  {
    name: 'list_entities',
    description: "Top entities of a kind by mention count. Use for 'who does he mention most', 'which places appear most', etc.",
    input_schema: {
      type: 'object',
      properties: {
        kind: { type: 'string', enum: ['all', 'person', 'work', 'place', 'theme', 'org'], default: 'all' },
        min_mentions: { type: 'integer', default: 2, minimum: 0 },
        limit: { type: 'integer', default: 30, minimum: 1, maximum: 100 },
      },
    },
  },
  {
    name: 'posts_about',
    description: 'Every post mentioning the named entity. Lookup is diacritic-insensitive.',
    input_schema: {
      type: 'object',
      properties: {
        entity_name: { type: 'string' },
        kind: { type: 'string', enum: ['person', 'work', 'place', 'theme', 'org'] },
        limit: { type: 'integer', default: 20, minimum: 1, maximum: 50 },
      },
      required: ['entity_name'],
    },
  },
  {
    name: 'entity_neighbors',
    description: "Entities that co-occur with the given one across posts. Use for 'what topics come up with X', 'who appears alongside Y'.",
    input_schema: {
      type: 'object',
      properties: {
        entity_name: { type: 'string' },
        kind: { type: 'string', enum: ['person', 'work', 'place', 'theme', 'org'] },
        limit: { type: 'integer', default: 15, minimum: 1, maximum: 50 },
      },
      required: ['entity_name'],
    },
  },
  {
    name: 'similar_posts',
    description: "Posts most similar to a given post via semantic embeddings. Use to answer 'what else did I write like this'.",
    input_schema: {
      type: 'object',
      properties: {
        post_id: { type: 'string' },
        limit: { type: 'integer', default: 8, minimum: 1, maximum: 20 },
      },
      required: ['post_id'],
    },
  },
  {
    name: 'corpus_stats',
    description: 'Summary of the archive: posts per blog, per language, entity counts.',
    input_schema: { type: 'object', properties: {} },
  },
];

function jsonResponse(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

export default async function handler(request) {
  if (request.method !== 'POST') {
    return jsonResponse(405, { error: 'method not allowed' });
  }

  const expectedHash = (process.env.WSB_SITE_PASSWORD_HASH || '').trim();
  if (expectedHash) {
    const given = (request.headers.get('X-Password-Hash') || '').trim();
    if (given !== expectedHash) {
      return jsonResponse(401, { error: 'unauthorized' });
    }
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return jsonResponse(500, { error: 'ANTHROPIC_API_KEY not configured' });
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return jsonResponse(400, { error: 'invalid JSON' });
  }

  if (!Array.isArray(body?.messages)) {
    return jsonResponse(400, { error: 'messages must be an array' });
  }

  const payload = {
    model: body.model || DEFAULT_MODEL,
    max_tokens: Math.min(body.max_tokens || MAX_TOKENS, 4096),
    system: SYSTEM_PROMPT,
    tools: TOOLS,
    messages: body.messages,
    stream: true,
  };

  const upstream = await fetch(ANTHROPIC_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': ANTHROPIC_VERSION,
    },
    body: JSON.stringify(payload),
  });

  if (!upstream.ok) {
    let detail;
    try { detail = await upstream.json(); }
    catch { detail = { error: upstream.statusText }; }
    return jsonResponse(upstream.status, detail);
  }

  return new Response(upstream.body, {
    status: 200,
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      'Connection': 'keep-alive',
    },
  });
}
