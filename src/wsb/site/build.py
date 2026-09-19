"""Build the deployable site — a self-contained graph + chat interface.

Emits everything Vercel needs at the repo root:
  public/index.html   — the app (graph + chat, password-gated)
  public/data.json    — baked corpus data the tools operate on
  api/chat.py         — serverless proxy that adds the ANTHROPIC_API_KEY
  vercel.json         — routing + Python runtime config

The client executes tools locally against public/data.json; only the LLM
call itself round-trips to the serverless function. That keeps the key
server-side without needing to move the corpus off the CDN.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from wsb.db import Store
from wsb.graph.export import build_graph_json

TEMPLATE_DIR = Path(__file__).parent
TEMPLATE_HTML = TEMPLATE_DIR / "template.html"
API_TEMPLATE = TEMPLATE_DIR / "api_chat.js"
VERCEL_JSON = TEMPLATE_DIR / "vercel.json"


def _body_preview(text: str | None, max_chars: int = 800) -> str:
    if not text:
        return ""
    return text[:max_chars]


def build_site_data(store: Store, min_similarity: float = 0.5) -> dict:
    """Extend the graph JSON with body previews so the chat tools have text
    to reason over. Everything else is reused from the graph exporter."""
    data = build_graph_json(store, min_similarity=min_similarity)
    body_by_id = {
        r["id"]: _body_preview(r["body_text"])
        for r in store.conn.execute("SELECT id, body_text FROM posts")
    }
    for post in data["posts"]:
        post["body_preview"] = body_by_id.get(post["id"], "")
    return data


def build_site(
    store: Store,
    output_root: Path,
    password: str | None = None,
    min_similarity: float = 0.5,
) -> dict:
    """Assemble the deploy tree at output_root. Vercel serves the root as
    static, and the `api/` folder as Python serverless functions. Everything
    the deploy needs is regenerated here.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    api = output_root / "api"
    api.mkdir(exist_ok=True)

    data = build_site_data(store, min_similarity=min_similarity)
    (output_root / "data.json").write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    )

    html = TEMPLATE_HTML.read_text()
    password_hash = ""
    if password:
        password_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    html = html.replace("__WSB_PASSWORD_HASH__", password_hash)
    (output_root / "index.html").write_text(html)

    # Ship the Edge Function; remove any older Python entrypoint so Vercel
    # doesn't try to serve both.
    shutil.copyfile(API_TEMPLATE, api / "chat.js")
    (api / "chat.py").unlink(missing_ok=True)
    shutil.copyfile(VERCEL_JSON, output_root / "vercel.json")

    return {
        "posts": len(data["posts"]),
        "entities": len(data["entities"]),
        "similar_edges": len(data["edges"]["similar"]),
        "data_size_kb": (output_root / "data.json").stat().st_size // 1024,
        "html_size_kb": (output_root / "index.html").stat().st_size // 1024,
        "password_gate": bool(password),
    }
