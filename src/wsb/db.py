"""SQLite storage — the source of truth for the whole pipeline.

Neo4j is a projection built from these tables and can be dropped and rebuilt
at any time. Keeping SQLite canonical means the archive survives even if you
never run a graph database, which matters for anyone you hand this to.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from wsb.models import BlogInfo, Edge, NormalizedPost, RawPost
from wsb.normalize import fold

SCHEMA = """
CREATE TABLE IF NOT EXISTS blogs (
    slug          TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    url           TEXT NOT NULL,
    platform      TEXT NOT NULL,
    default_lang  TEXT
);

CREATE TABLE IF NOT EXISTS posts (
    id             TEXT PRIMARY KEY,          -- {blog_slug}:{source_post_id}
    blog_slug      TEXT NOT NULL REFERENCES blogs(slug),
    source_post_id TEXT NOT NULL,
    url            TEXT,
    title          TEXT,
    published_at   TEXT,
    updated_at     TEXT,
    author         TEXT,
    lang           TEXT,
    body_md        TEXT,
    body_text      TEXT,
    search_text    TEXT,
    word_count     INTEGER DEFAULT 0,
    comment_count  INTEGER DEFAULT 0,
    content_hash   TEXT,
    raw_json       TEXT
);

CREATE INDEX IF NOT EXISTS idx_posts_blog ON posts(blog_slug);
CREATE INDEX IF NOT EXISTS idx_posts_date ON posts(published_at);

CREATE TABLE IF NOT EXISTS tags (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    normalized  TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS post_tags (
    post_id  TEXT NOT NULL REFERENCES posts(id),
    tag_id   INTEGER NOT NULL REFERENCES tags(id),
    PRIMARY KEY (post_id, tag_id)
);

CREATE TABLE IF NOT EXISTS media (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id  TEXT NOT NULL REFERENCES posts(id),
    kind     TEXT NOT NULL,
    url      TEXT NOT NULL,
    caption  TEXT
);

CREATE TABLE IF NOT EXISTS links (
    post_id  TEXT NOT NULL REFERENCES posts(id),
    url      TEXT NOT NULL
);

-- Populated by the enrichment stage (LLM extraction).
CREATE TABLE IF NOT EXISTS entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,      -- person | work | place | theme | org
    name        TEXT NOT NULL,
    normalized  TEXT NOT NULL,
    UNIQUE (kind, normalized)
);

CREATE TABLE IF NOT EXISTS post_entities (
    post_id     TEXT NOT NULL REFERENCES posts(id),
    entity_id   INTEGER NOT NULL REFERENCES entities(id),
    mention     TEXT,
    confidence  REAL DEFAULT 1.0,
    PRIMARY KEY (post_id, entity_id)
);

-- Tracks which posts have been through the LLM extractor and against which
-- content_hash. If a post is later re-fetched with new content, its hash
-- changes and `posts_needing_extraction` will surface it again.
CREATE TABLE IF NOT EXISTS extractions (
    post_id       TEXT PRIMARY KEY REFERENCES posts(id),
    content_hash  TEXT NOT NULL,
    extracted_at  TEXT NOT NULL
);

-- Surface forms that were merged into a canonical entity. Keeps the merge
-- reversible and lets future extractions of the same alias be redirected.
CREATE TABLE IF NOT EXISTS entity_aliases (
    entity_id   INTEGER NOT NULL REFERENCES entities(id),
    alias       TEXT NOT NULL,
    normalized  TEXT NOT NULL,
    kind        TEXT NOT NULL,
    PRIMARY KEY (kind, normalized)
);

CREATE TABLE IF NOT EXISTS edges (
    src_type    TEXT NOT NULL,
    src_id      TEXT NOT NULL,
    dst_type    TEXT NOT NULL,
    dst_id      TEXT NOT NULL,
    rel         TEXT NOT NULL,
    weight      REAL DEFAULT 1.0,
    source      TEXT NOT NULL DEFAULT 'structural',
    confidence  REAL DEFAULT 1.0,
    PRIMARY KEY (src_type, src_id, dst_type, dst_id, rel)
);

CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_type, src_id);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);

CREATE VIRTUAL TABLE IF NOT EXISTS posts_fts USING fts5(
    post_id UNINDEXED,
    title,
    search_text,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""


class Store:
    def __init__(self, path: str | Path = "second_brain.db"):
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # ---------- writes ----------

    def upsert_blog(self, blog: BlogInfo) -> None:
        self.conn.execute(
            """INSERT INTO blogs (slug, name, url, platform, default_lang)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(slug) DO UPDATE SET
                 name=excluded.name, url=excluded.url,
                 platform=excluded.platform, default_lang=excluded.default_lang""",
            (blog.slug, blog.name, blog.url, blog.platform, blog.default_lang),
        )
        self.conn.commit()

    def post_id(self, blog_slug: str, source_post_id: str) -> str:
        return f"{blog_slug}:{source_post_id}"

    def upsert_post(self, blog_slug: str, raw: RawPost, norm: NormalizedPost) -> str:
        pid = self.post_id(blog_slug, raw.source_post_id)
        self.conn.execute(
            """INSERT INTO posts (id, blog_slug, source_post_id, url, title,
                   published_at, updated_at, author, lang, body_md, body_text,
                   search_text, word_count, comment_count, content_hash, raw_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 title=excluded.title, body_md=excluded.body_md,
                 body_text=excluded.body_text, search_text=excluded.search_text,
                 lang=excluded.lang, word_count=excluded.word_count,
                 comment_count=excluded.comment_count,
                 updated_at=excluded.updated_at, content_hash=excluded.content_hash""",
            (
                pid, blog_slug, raw.source_post_id, raw.url, raw.title,
                raw.published_at.isoformat() if raw.published_at else None,
                raw.updated_at.isoformat() if raw.updated_at else None,
                raw.author, norm.lang, norm.body_md, norm.body_text,
                norm.search_text, norm.word_count, raw.comment_count,
                raw.content_hash(), json.dumps(raw.raw, default=str),
            ),
        )

        self.conn.execute("DELETE FROM posts_fts WHERE post_id = ?", (pid,))
        self.conn.execute(
            "INSERT INTO posts_fts (post_id, title, search_text) VALUES (?,?,?)",
            (pid, fold(raw.title), norm.search_text),
        )

        self.conn.execute("DELETE FROM media WHERE post_id = ?", (pid,))
        self.conn.executemany(
            "INSERT INTO media (post_id, kind, url, caption) VALUES (?,?,?,?)",
            [(pid, m.kind, m.url, m.caption) for m in norm.media],
        )

        self.conn.execute("DELETE FROM links WHERE post_id = ?", (pid,))
        self.conn.executemany(
            "INSERT INTO links (post_id, url) VALUES (?,?)",
            [(pid, url) for url in norm.outbound_links],
        )

        self.conn.execute("DELETE FROM post_tags WHERE post_id = ?", (pid,))
        for tag in raw.tags:
            normalized = fold(tag).strip()
            if not normalized:
                continue
            self.conn.execute(
                "INSERT OR IGNORE INTO tags (name, normalized) VALUES (?,?)",
                (tag, normalized),
            )
            row = self.conn.execute(
                "SELECT id FROM tags WHERE normalized = ?", (normalized,)
            ).fetchone()
            self.conn.execute(
                "INSERT OR IGNORE INTO post_tags (post_id, tag_id) VALUES (?,?)",
                (pid, row["id"]),
            )

        self.conn.commit()
        return pid

    def upsert_entity(self, kind: str, name: str) -> int:
        """Return the id of the entity, creating it if new.

        Merging happens on (kind, fold(name)) so diacritic variants collapse.
        """
        normalized = fold(name).strip()
        if not normalized:
            raise ValueError("entity name folds to empty string")
        self.conn.execute(
            "INSERT OR IGNORE INTO entities (kind, name, normalized) VALUES (?,?,?)",
            (kind, name, normalized),
        )
        row = self.conn.execute(
            "SELECT id FROM entities WHERE kind = ? AND normalized = ?",
            (kind, normalized),
        ).fetchone()
        return row["id"]

    def replace_post_entities(
        self, post_id: str, mentions: list[tuple[int, str, float]]
    ) -> None:
        """Swap the entity mentions for a post, leaving other posts untouched.

        `mentions` is (entity_id, surface_form, confidence).
        """
        self.conn.execute("DELETE FROM post_entities WHERE post_id = ?", (post_id,))
        self.conn.executemany(
            """INSERT OR REPLACE INTO post_entities
               (post_id, entity_id, mention, confidence) VALUES (?,?,?,?)""",
            [(post_id, eid, mention, conf) for eid, mention, conf in mentions],
        )

    def merge_entities(self, canonical_id: int, member_ids: list[int]) -> int:
        """Redirect all mentions of `member_ids` to `canonical_id`, then drop
        the member entity rows. Original names are preserved in entity_aliases
        so the merge is auditable and reversible.

        Returns the number of post-entity mentions redirected.
        """
        if canonical_id in member_ids:
            member_ids = [m for m in member_ids if m != canonical_id]
        if not member_ids:
            return 0

        placeholders = ",".join("?" * len(member_ids))

        # Snapshot members before mutation so we can write aliases.
        members = self.conn.execute(
            f"SELECT id, kind, name, normalized FROM entities WHERE id IN ({placeholders})",
            member_ids,
        ).fetchall()

        for m in members:
            self.conn.execute(
                """INSERT OR IGNORE INTO entity_aliases (entity_id, alias, normalized, kind)
                   VALUES (?, ?, ?, ?)""",
                (canonical_id, m["name"], m["normalized"], m["kind"]),
            )

        # Repoint post_entities. Where a post already links to canonical AND a
        # member, keep the highest confidence and let INSERT OR REPLACE win.
        member_mentions = self.conn.execute(
            f"""SELECT post_id, entity_id, mention, confidence
                FROM post_entities WHERE entity_id IN ({placeholders})""",
            member_ids,
        ).fetchall()

        self.conn.execute(
            f"DELETE FROM post_entities WHERE entity_id IN ({placeholders})",
            member_ids,
        )

        redirected = 0
        for row in member_mentions:
            existing = self.conn.execute(
                "SELECT confidence FROM post_entities WHERE post_id = ? AND entity_id = ?",
                (row["post_id"], canonical_id),
            ).fetchone()
            if existing and existing["confidence"] >= row["confidence"]:
                continue
            self.conn.execute(
                """INSERT OR REPLACE INTO post_entities
                   (post_id, entity_id, mention, confidence) VALUES (?,?,?,?)""",
                (row["post_id"], canonical_id, row["mention"], row["confidence"]),
            )
            redirected += 1

        self.conn.execute(
            f"DELETE FROM entities WHERE id IN ({placeholders})",
            member_ids,
        )
        self.conn.commit()
        return redirected

    def entities_by_kind(self, kind: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT e.id, e.kind, e.name, e.normalized,
                      COUNT(pe.post_id) AS mentions
               FROM entities e
               LEFT JOIN post_entities pe ON pe.entity_id = e.id
               WHERE e.kind = ?
               GROUP BY e.id
               ORDER BY mentions DESC, e.name""",
            (kind,),
        ).fetchall()

    def mark_extracted(self, post_id: str, content_hash: str) -> None:
        self.conn.execute(
            """INSERT INTO extractions (post_id, content_hash, extracted_at)
               VALUES (?, ?, datetime('now'))
               ON CONFLICT(post_id) DO UPDATE SET
                 content_hash = excluded.content_hash,
                 extracted_at = excluded.extracted_at""",
            (post_id, content_hash),
        )
        self.conn.commit()

    def posts_needing_extraction(
        self, blog: str | None = None, limit: int | None = None, force: bool = False
    ) -> list[sqlite3.Row]:
        """Posts whose content_hash has no matching extraction record.

        `force=True` returns every post regardless of extraction state.
        """
        sql = """SELECT p.id, p.blog_slug, p.title, p.lang, p.body_text,
                        p.content_hash, GROUP_CONCAT(t.name, '||') AS tags
                 FROM posts p
                 LEFT JOIN post_tags pt ON pt.post_id = p.id
                 LEFT JOIN tags t ON t.id = pt.tag_id"""
        clauses = []
        params: list = []
        if blog:
            clauses.append("p.blog_slug = ?")
            params.append(blog)
        if not force:
            clauses.append(
                "NOT EXISTS (SELECT 1 FROM extractions x "
                "WHERE x.post_id = p.id AND x.content_hash = p.content_hash)"
            )
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " GROUP BY p.id ORDER BY p.published_at DESC"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return self.conn.execute(sql, params).fetchall()

    def replace_edges(self, source: str, edges: list[Edge]) -> int:
        """Swap out every edge from one provenance tier, leaving others intact."""
        self.conn.execute("DELETE FROM edges WHERE source = ?", (source,))
        self.conn.executemany(
            """INSERT OR REPLACE INTO edges
               (src_type, src_id, dst_type, dst_id, rel, weight, source, confidence)
               VALUES (?,?,?,?,?,?,?,?)""",
            [
                (e.src_type, e.src_id, e.dst_type, e.dst_id, e.rel,
                 e.weight, e.source, e.confidence)
                for e in edges
            ],
        )
        self.conn.commit()
        return len(edges)

    # ---------- reads ----------

    def search(self, query: str, limit: int = 20) -> list[sqlite3.Row]:
        """Full-text search across every blog, diacritic-insensitive."""
        return self.conn.execute(
            """SELECT p.id, p.title, p.url, p.blog_slug, p.published_at, p.lang,
                      snippet(posts_fts, 2, '[', ']', ' … ', 12) AS snip,
                      bm25(posts_fts) AS score
               FROM posts_fts
               JOIN posts p ON p.id = posts_fts.post_id
               WHERE posts_fts MATCH ?
               ORDER BY score
               LIMIT ?""",
            (fold(query), limit),
        ).fetchall()

    def stats(self) -> dict:
        c = self.conn
        return {
            "blogs": c.execute("SELECT COUNT(*) n FROM blogs").fetchone()["n"],
            "posts": c.execute("SELECT COUNT(*) n FROM posts").fetchone()["n"],
            "tags": c.execute("SELECT COUNT(*) n FROM tags").fetchone()["n"],
            "media": c.execute("SELECT COUNT(*) n FROM media").fetchone()["n"],
            "edges": c.execute("SELECT COUNT(*) n FROM edges").fetchone()["n"],
            "entities": c.execute("SELECT COUNT(*) n FROM entities").fetchone()["n"],
            "by_blog": [
                dict(r) for r in c.execute(
                    """SELECT b.slug, b.name, COUNT(p.id) posts,
                              MIN(p.published_at) first, MAX(p.published_at) last
                       FROM blogs b LEFT JOIN posts p ON p.blog_slug = b.slug
                       GROUP BY b.slug ORDER BY posts DESC"""
                ).fetchall()
            ],
            "by_lang": [
                dict(r) for r in c.execute(
                    "SELECT lang, COUNT(*) n FROM posts GROUP BY lang ORDER BY n DESC"
                ).fetchall()
            ],
        }
