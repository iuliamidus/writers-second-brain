"""MCP server — the writer's interface to the archive.

Read-only by design. Everything the chatbot can do goes through SQLite (the
canonical store); Neo4j is only touched by `run_cypher`, and only in read
transactions. This means the archive stays queryable even when the graph
database is stopped.

Handlers are exposed as plain functions so tests can exercise them without
setting up the MCP transport.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from wsb.normalize import fold


DEFAULT_DB = os.getenv("WSB_DB", "second_brain.db")


# ---------- SQLite helpers ----------

def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _rows_to_dicts(rows) -> list[dict]:
    return [dict(r) for r in rows]


# ---------- Tool handlers ----------

def search_posts(db_path: str, query: str, limit: int = 15) -> list[dict]:
    """Diacritic-insensitive full-text search across every blog."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """SELECT p.id, p.title, p.url, p.blog_slug, p.published_at, p.lang,
                      snippet(posts_fts, 2, '[', ']', ' … ', 12) AS snippet,
                      bm25(posts_fts) AS score
               FROM posts_fts
               JOIN posts p ON p.id = posts_fts.post_id
               WHERE posts_fts MATCH ?
               ORDER BY score
               LIMIT ?""",
            (fold(query), max(1, min(limit, 50))),
        ).fetchall()
        return _rows_to_dicts(rows)
    finally:
        conn.close()


def get_post(db_path: str, post_id: str) -> dict | None:
    """Full post detail: metadata, body, tags, extracted entities."""
    conn = _connect(db_path)
    try:
        post = conn.execute(
            """SELECT id, blog_slug, title, url, published_at, lang, author,
                      word_count, body_text
               FROM posts WHERE id = ?""",
            (post_id,),
        ).fetchone()
        if post is None:
            return None
        result = dict(post)
        result["tags"] = [
            r["name"] for r in conn.execute(
                """SELECT t.name FROM post_tags pt JOIN tags t ON t.id = pt.tag_id
                   WHERE pt.post_id = ? ORDER BY t.name""",
                (post_id,),
            )
        ]
        result["entities"] = [
            {"kind": r["kind"], "name": r["name"], "mention": r["mention"],
             "confidence": r["confidence"]}
            for r in conn.execute(
                """SELECT e.kind, e.name, pe.mention, pe.confidence
                   FROM post_entities pe JOIN entities e ON e.id = pe.entity_id
                   WHERE pe.post_id = ? ORDER BY e.kind, e.name""",
                (post_id,),
            )
        ]
        return result
    finally:
        conn.close()


VALID_KINDS = {"person", "work", "place", "theme", "org"}


def list_entities(
    db_path: str,
    kind: str = "all",
    min_mentions: int = 2,
    limit: int = 50,
) -> list[dict]:
    """Top entities of a given kind by mention count."""
    conn = _connect(db_path)
    try:
        sql = """SELECT e.kind, e.name, COUNT(pe.post_id) AS mentions
                 FROM entities e
                 LEFT JOIN post_entities pe ON pe.entity_id = e.id"""
        params: list = []
        if kind != "all":
            if kind not in VALID_KINDS:
                raise ValueError(f"kind must be one of {sorted(VALID_KINDS)} or 'all'")
            sql += " WHERE e.kind = ?"
            params.append(kind)
        sql += " GROUP BY e.id HAVING mentions >= ? ORDER BY mentions DESC, e.name LIMIT ?"
        params.extend([max(0, min_mentions), max(1, min(limit, 200))])
        return _rows_to_dicts(conn.execute(sql, params))
    finally:
        conn.close()


def posts_about(db_path: str, entity_name: str, kind: str | None = None,
                limit: int = 25) -> list[dict]:
    """Posts that mention the given entity. Entity lookup uses diacritic-fold
    so 'Bucuresti' matches 'București' and vice versa.
    """
    conn = _connect(db_path)
    try:
        normalized = fold(entity_name).strip()
        entity_sql = "SELECT id, kind, name FROM entities WHERE normalized = ?"
        params: list = [normalized]
        if kind:
            if kind not in VALID_KINDS:
                raise ValueError(f"kind must be one of {sorted(VALID_KINDS)}")
            entity_sql += " AND kind = ?"
            params.append(kind)
        entities = _rows_to_dicts(conn.execute(entity_sql, params))
        if not entities:
            return []

        entity_ids = [e["id"] for e in entities]
        placeholders = ",".join("?" * len(entity_ids))
        rows = conn.execute(
            f"""SELECT DISTINCT p.id, p.title, p.url, p.blog_slug,
                       p.published_at, p.lang
                FROM post_entities pe JOIN posts p ON p.id = pe.post_id
                WHERE pe.entity_id IN ({placeholders})
                ORDER BY p.published_at DESC
                LIMIT ?""",
            (*entity_ids, max(1, min(limit, 100))),
        ).fetchall()
        return _rows_to_dicts(rows)
    finally:
        conn.close()


def entity_neighbors(db_path: str, entity_name: str, kind: str | None = None,
                     limit: int = 20) -> list[dict]:
    """Entities that co-occur with the given entity across posts."""
    conn = _connect(db_path)
    try:
        normalized = fold(entity_name).strip()
        entity_sql = "SELECT id FROM entities WHERE normalized = ?"
        params: list = [normalized]
        if kind:
            if kind not in VALID_KINDS:
                raise ValueError(f"kind must be one of {sorted(VALID_KINDS)}")
            entity_sql += " AND kind = ?"
            params.append(kind)
        entities = conn.execute(entity_sql, params).fetchall()
        if not entities:
            return []

        entity_ids = [e["id"] for e in entities]
        placeholders = ",".join("?" * len(entity_ids))
        rows = conn.execute(
            f"""SELECT other.kind AS kind, other.name AS name,
                       COUNT(DISTINCT pe1.post_id) AS shared_posts
                FROM post_entities pe1
                JOIN post_entities pe2 ON pe2.post_id = pe1.post_id
                JOIN entities other ON other.id = pe2.entity_id
                WHERE pe1.entity_id IN ({placeholders})
                  AND pe2.entity_id NOT IN ({placeholders})
                GROUP BY other.id
                ORDER BY shared_posts DESC, other.name
                LIMIT ?""",
            (*entity_ids, *entity_ids, max(1, min(limit, 100))),
        ).fetchall()
        return _rows_to_dicts(rows)
    finally:
        conn.close()


def corpus_stats(db_path: str) -> dict:
    from wsb.db import Store
    store = Store(db_path)
    try:
        return store.stats()
    finally:
        store.close()


def run_cypher(query: str) -> dict:
    """Execute a read-only Cypher query against Neo4j.

    Neo4j is optional for the rest of the pipeline; if it is not reachable,
    the tool returns a clear error rather than crashing. Writes are rejected
    by using execute_read at the driver level.
    """
    try:
        from neo4j import GraphDatabase
        from neo4j.exceptions import Neo4jError, ServiceUnavailable
    except ImportError:
        return {"error": "neo4j driver not installed"}

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "secondbrain")

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
    except Exception as exc:
        return {"error": f"could not create driver: {exc}"}

    try:
        with driver.session() as session:
            try:
                result = session.execute_read(
                    lambda tx: [record.data() for record in tx.run(query)]
                )
            except Neo4jError as exc:
                return {"error": f"cypher error: {exc.message}"}
            except ServiceUnavailable as exc:
                return {"error": f"neo4j unreachable: {exc}"}
            return {"records": result, "count": len(result)}
    finally:
        driver.close()


# ---------- MCP wiring ----------

TOOL_SCHEMAS = [
    {
        "name": "search_posts",
        "description": (
            "Diacritic-insensitive full-text search across every blog. "
            "Use this to find posts by keyword. Returns id, title, url, "
            "blog_slug, published_at, lang, and a highlighted snippet."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (Romanian or English)"},
                "limit": {"type": "integer", "default": 15, "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_post",
        "description": (
            "Return full post detail: title, body, url, tags, and extracted "
            "entities. Use the id returned by search_posts."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"post_id": {"type": "string"}},
            "required": ["post_id"],
        },
    },
    {
        "name": "list_entities",
        "description": (
            "Top entities of a given kind (person, work, place, theme, org) "
            "by mention count. Use to answer 'who does the writer mention "
            "most', 'which places appear most often', etc."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["all", "person", "work", "place", "theme", "org"],
                    "default": "all",
                },
                "min_mentions": {"type": "integer", "default": 2, "minimum": 0},
                "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 200},
            },
        },
    },
    {
        "name": "posts_about",
        "description": (
            "Every post that mentions the named entity. Entity lookup is "
            "diacritic-insensitive so 'Bucuresti' matches 'București'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity_name": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": ["person", "work", "place", "theme", "org"],
                },
                "limit": {"type": "integer", "default": 25, "minimum": 1, "maximum": 100},
            },
            "required": ["entity_name"],
        },
    },
    {
        "name": "entity_neighbors",
        "description": (
            "Entities that co-occur with the given entity across posts. Use "
            "to explore 'what topics come up with X' or 'which people appear "
            "alongside Y'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity_name": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": ["person", "work", "place", "theme", "org"],
                },
                "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
            },
            "required": ["entity_name"],
        },
    },
    {
        "name": "corpus_stats",
        "description": "Summary of the corpus: posts per blog, per language, entity counts, edge counts.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_cypher",
        "description": (
            "Execute a read-only Cypher query against Neo4j. Advanced tool. "
            "Writes are rejected. Returns JSON records or an error message. "
            "Use for structural questions the other tools cannot answer."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]


async def handle_call(name: str, arguments: dict[str, Any], db_path: str) -> Any:
    """Dispatch a tool call by name. Kept sync-ish since our work is I/O
    bound on SQLite/Neo4j and we do not gain from real async here.
    """
    args = arguments or {}
    if name == "search_posts":
        return search_posts(db_path, args["query"], args.get("limit", 15))
    if name == "get_post":
        return get_post(db_path, args["post_id"])
    if name == "list_entities":
        return list_entities(
            db_path,
            args.get("kind", "all"),
            args.get("min_mentions", 2),
            args.get("limit", 50),
        )
    if name == "posts_about":
        return posts_about(
            db_path, args["entity_name"], args.get("kind"), args.get("limit", 25)
        )
    if name == "entity_neighbors":
        return entity_neighbors(
            db_path, args["entity_name"], args.get("kind"), args.get("limit", 20)
        )
    if name == "corpus_stats":
        return corpus_stats(db_path)
    if name == "run_cypher":
        return run_cypher(args["query"])
    raise ValueError(f"unknown tool: {name}")


async def serve(db_path: str = DEFAULT_DB) -> None:
    """Run the stdio MCP server. Blocks until the client disconnects."""
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    db_path = str(Path(db_path).resolve())
    server = Server("wsb")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [Tool(**schema) for schema in TOOL_SCHEMAS]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
        try:
            result = await handle_call(name, arguments or {}, db_path)
        except Exception as exc:
            result = {"error": str(exc), "tool": name}
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2, default=str))]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
