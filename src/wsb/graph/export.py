"""Export the graph as a self-contained interactive HTML file.

The point is to hand something to a non-technical reader — no CLI, no
Neo4j, no login. They double-click the file and the whole corpus is a
graph they can filter, zoom and pan. All data is embedded in a single
JSON blob inside a `<script>` tag; cytoscape.js loads from CDN.

Only fields the reader is meant to see are exported. There is no API
key, session token or private path in the resulting HTML; the archive
itself is the writer's own material.
"""

from __future__ import annotations

import json
from pathlib import Path

from wsb.db import Store


TEMPLATE_PATH = Path(__file__).parent / "graph_template.html"


def build_graph_json(store: Store, min_similarity: float = 0.5) -> dict:
    """Assemble the nodes/edges dict the HTML template consumes."""
    conn = store.conn

    blogs = [dict(r) for r in conn.execute(
        "SELECT slug, name, url, platform FROM blogs"
    )]

    tags_by_post: dict[str, list[str]] = {}
    for r in conn.execute(
        """SELECT pt.post_id, t.name FROM post_tags pt
           JOIN tags t ON t.id = pt.tag_id"""
    ):
        tags_by_post.setdefault(r["post_id"], []).append(r["name"])

    entities_by_post: dict[str, list[dict]] = {}
    for r in conn.execute(
        """SELECT pe.post_id, e.kind, e.name FROM post_entities pe
           JOIN entities e ON e.id = pe.entity_id"""
    ):
        entities_by_post.setdefault(r["post_id"], []).append(
            {"kind": r["kind"], "name": r["name"]}
        )

    posts = [
        {
            "id": r["id"],
            "title": r["title"] or "(untitled)",
            "blog": r["blog_slug"],
            "date": (r["published_at"] or "")[:10],
            "year": int(r["published_at"][:4]) if r["published_at"] else None,
            "lang": r["lang"],
            "url": r["url"],
            "tags": tags_by_post.get(r["id"], []),
            "entities": entities_by_post.get(r["id"], []),
        }
        for r in conn.execute(
            """SELECT id, title, blog_slug, published_at, lang, url
               FROM posts"""
        )
    ]

    entities = [
        {
            "id": f"entity:{r['kind']}:{r['normalized']}",
            "kind": r["kind"],
            "name": r["name"],
        }
        for r in conn.execute("SELECT kind, name, normalized FROM entities")
    ]

    similar_edges = [
        {
            "source": r["src_id"],
            "target": r["dst_id"],
            "weight": float(r["weight"]),
            "kind": "similar",
        }
        for r in conn.execute(
            """SELECT src_id, dst_id, weight FROM edges
               WHERE rel = 'SIMILAR_TO' AND weight >= ?""",
            (min_similarity,),
        )
    ]

    tagged_edges = [
        {
            "source": r["src_id"],
            "target": r["dst_id"],
            "kind": "tagged",
        }
        for r in conn.execute(
            "SELECT src_id, dst_id FROM edges WHERE rel = 'TAGGED'"
        )
    ]

    mentions_edges = [
        {
            "source": r["src_id"],
            "target": f"entity:{r['dst_id']}",
            "kind": "mentions",
        }
        for r in conn.execute(
            "SELECT src_id, dst_id FROM edges WHERE rel = 'MENTIONS'"
        )
    ]

    published_edges = [
        {"source": r["src_id"], "target": r["dst_id"], "kind": "published"}
        for r in conn.execute(
            "SELECT src_id, dst_id FROM edges WHERE rel = 'PUBLISHED_ON'"
        )
    ]

    years = sorted({p["year"] for p in posts if p["year"] is not None})

    return {
        "blogs": blogs,
        "posts": posts,
        "entities": entities,
        "edges": {
            "similar": similar_edges,
            "tagged": tagged_edges,
            "mentions": mentions_edges,
            "published": published_edges,
        },
        "meta": {
            "min_similarity": min_similarity,
            "year_min": years[0] if years else None,
            "year_max": years[-1] if years else None,
        },
    }


def render_html(graph: dict, template_path: Path | None = None) -> str:
    """Embed the graph JSON inside the template and return the full HTML."""
    template = (template_path or TEMPLATE_PATH).read_text()
    payload = json.dumps(graph, ensure_ascii=False, separators=(",", ":"))
    # `</script>` inside a data payload would end the script tag early.
    payload = payload.replace("</", "<\\/")
    return template.replace("__WSB_GRAPH_JSON__", payload)
