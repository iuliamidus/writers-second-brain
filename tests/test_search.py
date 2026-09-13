from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from wsb.db import Store
from wsb.models import BlogInfo, Edge, RawPost
from wsb.normalize import normalize
from wsb.search import (
    fts_search,
    hybrid_search,
    rrf_fuse,
    semantic_search,
    similar_posts,
)


class _FakeEmbedder:
    """Returns a fixed vector per input string. Tests inject the mapping."""

    def __init__(self, mapping: dict[str, np.ndarray]):
        self.mapping = mapping

    def encode(self, texts):
        vectors = []
        for t in texts:
            if t not in self.mapping:
                raise KeyError(f"no vector for {t!r}")
            v = self.mapping[t].astype(np.float32)
            v /= np.linalg.norm(v) + 1e-12
            vectors.append(v)
        return np.vstack(vectors)


def _make_store(tmp_path, posts):
    store = Store(tmp_path / "t.db")
    for slug in {p["blog"] for p in posts}:
        store.upsert_blog(BlogInfo(slug=slug, name=slug, url=f"http://{slug}", platform="rss"))
    ids = []
    for p in posts:
        raw = RawPost(
            source_post_id=p["sid"],
            url=f"http://{p['blog']}/{p['sid']}",
            title=p["title"],
            body_html=f"<p>{p['body']}</p>",
            published_at=datetime(2024, 1, 1),
        )
        ids.append(store.upsert_post(p["blog"], raw, normalize(raw)))
    return store, ids


# ---------- RRF ----------

def test_rrf_fuse_boosts_docs_that_rank_high_in_both():
    a = [{"id": "x"}, {"id": "y"}, {"id": "z"}]
    b = [{"id": "y"}, {"id": "x"}, {"id": "w"}]
    fused = rrf_fuse([a, b], limit=10)
    ranked_ids = [d["id"] for d in fused]
    # y ranks 2nd + 1st, x ranks 1st + 2nd — both should beat z (only in a) and w (only in b).
    assert set(ranked_ids[:2]) == {"x", "y"}
    assert "z" in ranked_ids and "w" in ranked_ids


def test_rrf_fuse_respects_limit():
    a = [{"id": str(i)} for i in range(30)]
    b = [{"id": str(i)} for i in range(30)]
    assert len(rrf_fuse([a, b], limit=5)) == 5


def test_rrf_fuse_carries_original_details():
    a = [{"id": "x", "title": "A title"}]
    b = [{"id": "x", "title": "A different title", "extra": "stuff"}]
    fused = rrf_fuse([a, b])
    assert fused[0]["title"] == "A title"  # first ranking wins on details
    assert "rrf_score" in fused[0]


# ---------- FTS ----------

def test_fts_search_returns_hits(tmp_path):
    store, ids = _make_store(tmp_path, [
        {"blog": "b", "sid": "1", "title": "About Norway",
         "body": "The fjords of Norway are wonderful, fjords."},
        {"blog": "b", "sid": "2", "title": "About Bucharest",
         "body": "Walking through Bucharest is a joy."},
    ])
    results = fts_search(store.conn, "Norway", limit=10)
    assert any(r["id"] == ids[0] for r in results)
    store.close()


# ---------- semantic ----------

def test_semantic_search_ranks_by_cosine(tmp_path):
    """Two posts, one whose embedding matches the query, one that doesn't."""
    store, ids = _make_store(tmp_path, [
        {"blog": "b", "sid": "1", "title": "About Norway",
         "body": "fjords fjords fjords fjords fjords fjords fjords fjords"},
        {"blog": "b", "sid": "2", "title": "About cooking",
         "body": "recipes recipes recipes recipes recipes recipes recipes"},
    ])
    q_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    norway_vec = np.array([0.95, 0.31, 0.0, 0.0], dtype=np.float32)
    cooking_vec = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    for pid, vec in zip(ids, [norway_vec, cooking_vec]):
        v = vec / (np.linalg.norm(vec) + 1e-12)
        store.save_embedding(pid, "test", v)

    embedder = _FakeEmbedder({"fjords": q_vec})
    results = semantic_search(store.conn, embedder, "fjords", "test", limit=10)
    assert results[0]["id"] == ids[0]  # Norway wins
    assert results[0]["cosine"] > results[1]["cosine"]
    store.close()


def test_semantic_search_empty_when_no_embeddings(tmp_path):
    store, _ = _make_store(tmp_path, [{"blog": "b", "sid": "1", "title": "T", "body": "body"}])
    embedder = _FakeEmbedder({"anything": np.array([1.0], dtype=np.float32)})
    assert semantic_search(store.conn, embedder, "anything", "test", limit=5) == []
    store.close()


# ---------- hybrid ----------

def test_hybrid_search_returns_both_kinds_of_hits(tmp_path):
    store, ids = _make_store(tmp_path, [
        {"blog": "b", "sid": "1", "title": "About Norway",
         "body": "The fjords are wonderful. Word occurs many times: keyword keyword keyword."},
        {"blog": "b", "sid": "2", "title": "About cooking",
         "body": "recipes and pans and stoves"},
    ])
    q_vec = np.array([1.0, 0.0], dtype=np.float32)
    for pid, vec in zip(ids, [np.array([1.0, 0.0]), np.array([0.0, 1.0])]):
        v = vec / (np.linalg.norm(vec) + 1e-12)
        store.save_embedding(pid, "test", v.astype(np.float32))

    embedder = _FakeEmbedder({"keyword": q_vec})
    results = hybrid_search(store.conn, embedder, "keyword", "test", limit=10)
    ids_found = [r["id"] for r in results]
    assert ids[0] in ids_found
    assert all("rrf_score" in r for r in results)
    store.close()


# ---------- similar_posts ----------

def test_similar_posts_reads_statistical_edges(tmp_path):
    store, ids = _make_store(tmp_path, [
        {"blog": "b", "sid": str(i), "title": f"P{i}", "body": "body"} for i in range(4)
    ])
    edges = [
        Edge("Post", ids[0], "Post", ids[1], "SIMILAR_TO",
             weight=0.9, source="statistical", confidence=0.9),
        Edge("Post", ids[0], "Post", ids[2], "SIMILAR_TO",
             weight=0.7, source="statistical", confidence=0.7),
        Edge("Post", ids[2], "Post", ids[3], "SIMILAR_TO",
             weight=0.6, source="statistical", confidence=0.6),
    ]
    store.replace_edges("statistical", edges)

    result = similar_posts(store.conn, ids[0], limit=10)
    result_ids = {r["id"] for r in result}
    assert result_ids == {ids[1], ids[2]}
    # Ordered by weight descending.
    assert result[0]["id"] == ids[1]
    assert result[0]["similarity"] == pytest.approx(0.9)
    store.close()


def test_similar_posts_returns_empty_when_no_edges(tmp_path):
    store, ids = _make_store(tmp_path, [{"blog": "b", "sid": "1", "title": "T", "body": "b"}])
    assert similar_posts(store.conn, ids[0]) == []
    store.close()


def test_similar_posts_works_for_edge_in_either_direction(tmp_path):
    store, ids = _make_store(tmp_path, [
        {"blog": "b", "sid": "1", "title": "A", "body": "b"},
        {"blog": "b", "sid": "2", "title": "B", "body": "b"},
    ])
    store.replace_edges("statistical", [
        Edge("Post", ids[0], "Post", ids[1], "SIMILAR_TO",
             weight=0.8, source="statistical", confidence=0.8),
    ])
    # Same edge should surface for both endpoints.
    from_a = similar_posts(store.conn, ids[0])
    from_b = similar_posts(store.conn, ids[1])
    assert [r["id"] for r in from_a] == [ids[1]]
    assert [r["id"] for r in from_b] == [ids[0]]
    store.close()
