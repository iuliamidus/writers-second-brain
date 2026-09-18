"""Vercel serverless proxy: adds ANTHROPIC_API_KEY, forwards to the Messages
API, returns the response verbatim.

Client executes tool calls locally against public/data.json — so this
function never touches the corpus, only the LLM. Kept intentionally small.
"""

from __future__ import annotations

import json
import os
import urllib.request
from http.server import BaseHTTPRequestHandler


ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 2048


SYSTEM_PROMPT = """You are a research companion for a writer with four blogs and 706 posts spanning 2011 to today, mixed Romanian and English. Your job is to help the writer see the shape of the archive: recurring themes, cross-blog echoes, forgotten posts.

Use the tools to look things up rather than guessing. The archive is bilingual — respond in the language the user writes to you in. If the user writes in Romanian, respond in Romanian; if in English, respond in English.

Post ids are namespaced as {blog_slug}:{source_id}. Get them from search_posts before calling similar_posts or get_post. Titles alone are not ids.

When you cite posts, include the title, date, and blog. When appropriate, format post titles as clickable links using the url field returned by the tools.

Prefer concise, structural answers over long summaries. The user is a writer who values good prose."""


TOOLS = [
    {
        "name": "search_posts",
        "description": "Diacritic-insensitive keyword search across every blog. Returns id, title, url, blog, date, lang, and a highlighted snippet. Use to find posts by keyword.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 15, "minimum": 1, "maximum": 30},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_post",
        "description": "Full detail for a single post: title, body preview, url, tags, mentioned entities. Use the id from search_posts.",
        "input_schema": {
            "type": "object",
            "properties": {"post_id": {"type": "string"}},
            "required": ["post_id"],
        },
    },
    {
        "name": "list_entities",
        "description": "Top entities of a kind by mention count. Use for 'who does he mention most', 'which places appear most', etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["all", "person", "work", "place", "theme", "org"],
                    "default": "all",
                },
                "min_mentions": {"type": "integer", "default": 2, "minimum": 0},
                "limit": {"type": "integer", "default": 30, "minimum": 1, "maximum": 100},
            },
        },
    },
    {
        "name": "posts_about",
        "description": "Every post mentioning the named entity. Lookup is diacritic-insensitive.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_name": {"type": "string"},
                "kind": {"type": "string", "enum": ["person", "work", "place", "theme", "org"]},
                "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 50},
            },
            "required": ["entity_name"],
        },
    },
    {
        "name": "entity_neighbors",
        "description": "Entities that co-occur with the given one across posts. Use for 'what topics come up with X', 'who appears alongside Y'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_name": {"type": "string"},
                "kind": {"type": "string", "enum": ["person", "work", "place", "theme", "org"]},
                "limit": {"type": "integer", "default": 15, "minimum": 1, "maximum": 50},
            },
            "required": ["entity_name"],
        },
    },
    {
        "name": "similar_posts",
        "description": "Posts most similar to a given post via semantic embeddings. Use to answer 'what else did I write like this'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "post_id": {"type": "string"},
                "limit": {"type": "integer", "default": 8, "minimum": 1, "maximum": 20},
            },
            "required": ["post_id"],
        },
    },
    {
        "name": "corpus_stats",
        "description": "Summary of the archive: posts per blog, per language, entity counts.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def _json_response(handler: BaseHTTPRequestHandler, status: int, body: dict) -> None:
    payload = json.dumps(body).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


class handler(BaseHTTPRequestHandler):  # noqa: N801 — Vercel expects lowercase `handler`
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            return _json_response(self, 400, {"error": "invalid JSON"})

        expected_hash = os.environ.get("WSB_SITE_PASSWORD_HASH", "").strip()
        if expected_hash:
            given = self.headers.get("X-Password-Hash", "").strip()
            if given != expected_hash:
                return _json_response(self, 401, {"error": "unauthorized"})

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return _json_response(self, 500, {"error": "ANTHROPIC_API_KEY not configured"})

        messages = body.get("messages")
        if not isinstance(messages, list):
            return _json_response(self, 400, {"error": "messages must be an array"})

        payload = {
            "model": body.get("model", DEFAULT_MODEL),
            "max_tokens": min(int(body.get("max_tokens", MAX_TOKENS)), 4096),
            "system": SYSTEM_PROMPT,
            "tools": TOOLS,
            "messages": messages,
        }

        req = urllib.request.Request(
            ANTHROPIC_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_VERSION,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return _json_response(self, 200, data)
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode("utf-8"))
            except Exception:
                detail = {"error": exc.reason}
            return _json_response(self, exc.code, detail)
        except Exception as exc:  # network, timeout, etc.
            return _json_response(self, 502, {"error": str(exc)})

    def log_message(self, format, *args):  # silence Vercel's default logging
        return
