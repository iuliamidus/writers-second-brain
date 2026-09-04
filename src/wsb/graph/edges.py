"""Structural edges — derived from platform metadata, no LLM, no embeddings.

Build and query this tier before reaching for anything cleverer. Tags, dates
and self-links carry more signal than people expect, and every edge here is a
fact rather than an inference.
"""

from __future__ import annotations

import re
from collections import defaultdict

from wsb.db import Store
from wsb.models import Edge

# Matches series markers writers use without thinking: trailing "- 2", "(3)",
# "part 2", "#2". Catches things like "Norway – a Solitary Road Trip – 2".
SERIES_RE = re.compile(
    r"^(?P<stem>.+?)[\s\-–—(\[]*(?:part|partea|pt\.?|#)?\s*(?P<num>\d{1,2})\s*[)\]]?\s*$",
    re.IGNORECASE,
)


def _series_key(title: str) -> tuple[str, int] | None:
    match = SERIES_RE.match(title.strip())
    if not match:
        return None
    stem = match.group("stem").strip(" -–—([")
    if len(stem) < 8:  # too short to be a real series stem
        return None
    return stem.lower(), int(match.group("num"))


def build_structural_edges(store: Store) -> list[Edge]:
    conn = store.conn
    edges: list[Edge] = []

    # Post -> Blog
    for row in conn.execute("SELECT id, blog_slug FROM posts"):
        edges.append(Edge("Post", row["id"], "Blog", row["blog_slug"], "PUBLISHED_ON"))

    # Post -> Tag
    for row in conn.execute(
        """SELECT pt.post_id, t.normalized FROM post_tags pt
           JOIN tags t ON t.id = pt.tag_id"""
    ):
        edges.append(Edge("Post", row["post_id"], "Tag", row["normalized"], "TAGGED"))

    # Post -> Post, via the writer's own hyperlinks. Only internal links
    # resolve; everything else stays an outbound link and is ignored here.
    url_to_post = {
        row["url"].rstrip("/"): row["id"]
        for row in conn.execute("SELECT id, url FROM posts WHERE url IS NOT NULL")
    }
    for row in conn.execute("SELECT post_id, url FROM links"):
        target = url_to_post.get(row["url"].rstrip("/"))
        if target and target != row["post_id"]:
            edges.append(Edge("Post", row["post_id"], "Post", target, "LINKS_TO"))

    # Post -> Post, explicit series. The writer numbered these themselves, so
    # the ordering is authoritative rather than inferred.
    series: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in conn.execute("SELECT id, title FROM posts WHERE title IS NOT NULL"):
        key = _series_key(row["title"])
        if key:
            series[key[0]].append((key[1], row["id"]))

    for parts in series.values():
        if len(parts) < 2:
            continue
        parts.sort()
        for (_, earlier), (_, later) in zip(parts, parts[1:]):
            edges.append(Edge("Post", earlier, "Post", later, "CONTINUES", confidence=0.9))

    return edges


def build_entity_edges(store: Store) -> list[Edge]:
    """Post -> Entity edges from whatever the enrichment stage has extracted."""
    edges: list[Edge] = []
    for row in store.conn.execute(
        """SELECT pe.post_id, e.kind, e.normalized, pe.confidence
           FROM post_entities pe JOIN entities e ON e.id = pe.entity_id"""
    ):
        # Entity keys are namespaced by kind so that a band and a place with
        # the same name stay separate nodes.
        key = f"{row['kind']}:{row['normalized']}"
        edges.append(
            Edge("Post", row["post_id"], "Entity", key,
                 "MENTIONS", source="extracted", confidence=row["confidence"])
        )
    return edges
