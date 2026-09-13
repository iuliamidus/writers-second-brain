"""Retrieval — keyword, semantic, and the fusion of the two.

FTS5 finds posts that share words with the query. Dense cosine over BGE-M3
finds posts that share *meaning* — including across languages, since a
Romanian post about singurătate lands near an English post about solitude
in vector space even when the words never overlap.

Reciprocal rank fusion combines the two rankings by rank rather than by
score, which keeps the merge well-behaved when the two systems disagree on
absolute magnitudes. Callers who want just one flavour still can — call
`fts_search` or `semantic_search` directly.
"""

from __future__ import annotations

import sqlite3
from typing import Iterable

import numpy as np

from wsb.normalize import fold


RRF_CONSTANT = 60  # standard damping term from the RRF paper


def fts_search(conn: sqlite3.Connection, query: str, limit: int = 50) -> list[dict]:
    """Diacritic-insensitive FTS5 with BM25 ranking."""
    rows = conn.execute(
        """SELECT p.id, p.title, p.url, p.blog_slug, p.published_at, p.lang,
                  snippet(posts_fts, 2, '[', ']', ' … ', 12) AS snippet,
                  bm25(posts_fts) AS bm25_score
           FROM posts_fts
           JOIN posts p ON p.id = posts_fts.post_id
           WHERE posts_fts MATCH ?
           ORDER BY bm25_score
           LIMIT ?""",
        (fold(query), max(1, limit)),
    ).fetchall()
    return [dict(r) for r in rows]


def semantic_search(
    conn: sqlite3.Connection,
    embedder,
    query: str,
    model: str,
    limit: int = 50,
) -> list[dict]:
    """Encode the query, rank every embedded post by cosine similarity."""
    query_vec = embedder.encode([query])[0]
    query_vec = np.asarray(query_vec, dtype=np.float32).reshape(-1)

    ids: list[str] = []
    matrix_rows: list[np.ndarray] = []
    for row in conn.execute(
        "SELECT post_id, vector FROM embeddings WHERE model = ?",
        (model,),
    ):
        ids.append(row["post_id"])
        matrix_rows.append(np.frombuffer(row["vector"], dtype=np.float32))
    if not ids:
        return []

    matrix = np.vstack(matrix_rows)
    scores = matrix @ query_vec  # vectors are L2-normalised at store time
    take = min(limit, scores.shape[0])
    top_idx = np.argpartition(-scores, take - 1)[:take]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    top_ids = [ids[i] for i in top_idx]
    placeholders = ",".join("?" * len(top_ids))
    posts = conn.execute(
        f"""SELECT id, title, url, blog_slug, published_at, lang
            FROM posts WHERE id IN ({placeholders})""",
        top_ids,
    ).fetchall()
    id_to_post = {r["id"]: dict(r) for r in posts}

    out: list[dict] = []
    for i in top_idx:
        pid = ids[i]
        post = id_to_post.get(pid)
        if not post:
            continue
        out.append({**post, "cosine": float(scores[i])})
    return out


def rrf_fuse(
    rankings: Iterable[list[dict]],
    key: str = "id",
    c: int = RRF_CONSTANT,
    limit: int = 15,
) -> list[dict]:
    """Reciprocal rank fusion. Higher score = better; both rankings contribute
    1/(rank + c) per document. The first ranking wins tie-break on details."""
    scores: dict[str, float] = {}
    details: dict[str, dict] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            doc_id = doc.get(key)
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rank + 1 + c)
            details.setdefault(doc_id, doc)

    fused = [
        {**details[doc_id], "rrf_score": score}
        for doc_id, score in scores.items()
    ]
    fused.sort(key=lambda d: -d["rrf_score"])
    return fused[:limit]


def hybrid_search(
    conn: sqlite3.Connection,
    embedder,
    query: str,
    model: str,
    limit: int = 15,
    fts_pool: int = 50,
    semantic_pool: int = 50,
) -> list[dict]:
    """RRF over FTS5 and dense cosine rankings."""
    fts = fts_search(conn, query, limit=fts_pool)
    sem = semantic_search(conn, embedder, query, model, limit=semantic_pool)
    return rrf_fuse([fts, sem], limit=limit)


def similar_posts(
    conn: sqlite3.Connection,
    post_id: str,
    limit: int = 10,
) -> list[dict]:
    """Return posts similar to `post_id` via stored SIMILAR_TO edges. Reads
    the statistical tier, so this only returns results once `wsb graph build`
    has run against embeddings."""
    rows = conn.execute(
        """SELECT CASE WHEN e.src_id = ? THEN e.dst_id ELSE e.src_id END AS other_id,
                  e.weight AS similarity
           FROM edges e
           WHERE (e.src_id = ? OR e.dst_id = ?)
             AND e.rel = 'SIMILAR_TO'
           ORDER BY e.weight DESC
           LIMIT ?""",
        (post_id, post_id, post_id, max(1, limit)),
    ).fetchall()
    if not rows:
        return []

    other_ids = [r["other_id"] for r in rows]
    placeholders = ",".join("?" * len(other_ids))
    posts = conn.execute(
        f"""SELECT id, title, url, blog_slug, published_at, lang
            FROM posts WHERE id IN ({placeholders})""",
        other_ids,
    ).fetchall()
    id_to_post = {r["id"]: dict(r) for r in posts}

    out: list[dict] = []
    for row in rows:
        post = id_to_post.get(row["other_id"])
        if post:
            out.append({**post, "similarity": float(row["similarity"])})
    return out
