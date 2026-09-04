"""Project the SQLite tables into Neo4j.

This is deliberately one-directional and idempotent: Neo4j holds no state that
cannot be rebuilt. `wsb graph push --reset` wipes and reloads in seconds at
this corpus size, which means you can iterate on the schema freely.
"""

from __future__ import annotations

import os
import re

from neo4j import GraphDatabase

from wsb.db import Store

BATCH = 1000

CONSTRAINTS = [
    "CREATE CONSTRAINT post_id IF NOT EXISTS FOR (p:Post) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT blog_slug IF NOT EXISTS FOR (b:Blog) REQUIRE b.slug IS UNIQUE",
    "CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (t:Tag) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
    "CREATE FULLTEXT INDEX post_text IF NOT EXISTS FOR (p:Post) ON EACH [p.title, p.body]",
]


# Every node label we project, and the property its unique constraint covers.
# Adding a node type means adding it here and to CONSTRAINTS above.
NODE_KEYS = {
    "Post": "id",
    "Blog": "slug",
    "Tag": "name",
    "Entity": "key",
}

IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


def _check_identifier(name: str) -> str:
    """Guard anything interpolated into Cypher rather than parameterised."""
    if not IDENTIFIER_RE.match(name):
        raise ValueError(f"Refusing to interpolate unsafe Cypher identifier: {name!r}")
    return name


def _label_and_key(node_type: str) -> tuple[str, str]:
    if node_type not in NODE_KEYS:
        raise ValueError(
            f"Unknown node type {node_type!r}. Add it to NODE_KEYS and CONSTRAINTS."
        )
    return node_type, NODE_KEYS[node_type]


def _chunks(rows, size=BATCH):
    buf = []
    for row in rows:
        buf.append(row)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


class Neo4jLoader:
    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        self.driver = GraphDatabase.driver(
            uri or os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                user or os.getenv("NEO4J_USER", "neo4j"),
                password or os.getenv("NEO4J_PASSWORD", "secondbrain"),
            ),
        )

    def close(self) -> None:
        self.driver.close()

    def push(self, store: Store, reset: bool = True) -> dict:
        counts = {"blogs": 0, "posts": 0, "tags": 0, "entities": 0, "edges": 0}

        with self.driver.session() as session:
            if reset:
                session.run("MATCH (n) DETACH DELETE n")
            for stmt in CONSTRAINTS:
                session.run(stmt)

            blogs = [dict(r) for r in store.conn.execute("SELECT * FROM blogs")]
            session.run(
                """UNWIND $rows AS r
                   MERGE (b:Blog {slug: r.slug})
                   SET b.name = r.name, b.url = r.url, b.platform = r.platform""",
                rows=blogs,
            )
            counts["blogs"] = len(blogs)

            post_rows = store.conn.execute(
                """SELECT id, title, url, published_at, lang, word_count,
                          comment_count, blog_slug, substr(body_text, 1, 4000) AS body
                   FROM posts"""
            )
            for batch in _chunks(dict(r) for r in post_rows):
                session.run(
                    """UNWIND $rows AS r
                       MERGE (p:Post {id: r.id})
                       SET p.title = r.title,
                           p.url = r.url,
                           p.body = r.body,
                           p.lang = r.lang,
                           p.wordCount = r.word_count,
                           p.commentCount = r.comment_count,
                           p.publishedAt = CASE WHEN r.published_at IS NULL
                                                THEN NULL
                                                ELSE datetime(r.published_at) END,
                           p.year = CASE WHEN r.published_at IS NULL
                                         THEN NULL
                                         ELSE toInteger(substring(r.published_at, 0, 4)) END""",
                    rows=batch,
                )
                counts["posts"] += len(batch)

            tags = [dict(r) for r in store.conn.execute("SELECT name, normalized FROM tags")]
            session.run(
                """UNWIND $rows AS r
                   MERGE (t:Tag {name: r.normalized})
                   SET t.display = r.name""",
                rows=tags,
            )
            counts["tags"] = len(tags)

            entities = [
                dict(r) for r in store.conn.execute("SELECT kind, name, normalized FROM entities")
            ]
            if entities:
                session.run(
                    """UNWIND $rows AS r
                       MERGE (e:Entity {key: r.kind + ':' + r.normalized})
                       SET e.name = r.name, e.kind = r.kind, e.normalized = r.normalized""",
                    rows=entities,
                )
            counts["entities"] = len(entities)

            # Relationship type cannot be parameterised in Cypher, so we group
            # edges by rel and run one statement per type.
            rel_types = [
                (r["src_type"], r["dst_type"], r["rel"])
                for r in store.conn.execute(
                    "SELECT DISTINCT src_type, dst_type, rel FROM edges"
                )
            ]
            for src_type, dst_type, rel in rel_types:
                # Cypher cannot parameterise labels or relationship types, so
                # these three get interpolated. Validate them rather than
                # trusting the table: once an LLM is writing edge rows, this
                # is an injection path straight into the database.
                _check_identifier(rel)
                src_label, src_key = _label_and_key(src_type)
                dst_label, dst_key = _label_and_key(dst_type)

                rows = store.conn.execute(
                    """SELECT src_id, dst_id, weight, source, confidence
                       FROM edges WHERE src_type = ? AND dst_type = ? AND rel = ?""",
                    (src_type, dst_type, rel),
                )
                for batch in _chunks(dict(r) for r in rows):
                    # Both MATCHes carry a label and hit a unique constraint,
                    # so each is an index seek. Without the label Neo4j scans
                    # every node in the graph once per row.
                    session.run(
                        f"""UNWIND $rows AS r
                            MATCH (a:{src_label} {{{src_key}: r.src_id}})
                            MATCH (b:{dst_label} {{{dst_key}: r.dst_id}})
                            MERGE (a)-[e:{rel}]->(b)
                            SET e.weight = r.weight,
                                e.source = r.source,
                                e.confidence = r.confidence""",
                        rows=batch,
                    )
                    counts["edges"] += len(batch)

        return counts
