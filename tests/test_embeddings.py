from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from wsb.db import Store
from wsb.embeddings import Embedder, compose_from_row, compose_text
from wsb.graph.edges import build_similarity_edges
from wsb.models import BlogInfo, RawPost
from wsb.normalize import normalize


# ---------- compose_text ----------

def test_compose_text_joins_all_parts():
    text = compose_text(
        title="Amintiri",
        tags=["București", "poezie"],
        body="M-am plimbat prin oraș.",
        media_captions=["Turnul Bisericii Colțea"],
    )
    assert "Amintiri" in text
    assert "București" in text
    assert "poezie" in text
    assert "M-am plimbat" in text
    assert "Turnul" in text


def test_compose_text_survives_short_post_with_only_title_and_caption():
    """The whole point of composing: a two-word caption gets useful signal."""
    text = compose_text(
        title="Golden hour",
        tags=None,
        body="",
        media_captions=["Sunset over Trollstigen"],
    )
    assert "Golden hour" in text
    assert "Sunset over Trollstigen" in text


def test_compose_text_strips_empty_pieces():
    assert compose_text(None, None, None, None) == ""
    assert compose_text("  ", ["", "  "], "  ", [None, ""]) == ""


def test_compose_text_caps_body_length():
    text = compose_text("t", [], "x" * 10_000, [], max_body_chars=500)
    assert text.count("x") == 500


def test_compose_from_row_unpacks_pipe_delimited_fields():
    row = {
        "title": "T",
        "tags": "a||b||c",
        "body_text": "body",
        "captions": "cap1||cap2",
    }
    text = compose_from_row(row)
    assert "a, b, c" in text
    assert "cap1 · cap2" in text


def test_compose_from_row_handles_null_group_concat():
    row = {"title": "T", "tags": None, "body_text": "body", "captions": None}
    text = compose_from_row(row)
    assert "T" in text
    assert "body" in text
    assert "Tags:" not in text
    assert "Captions:" not in text


# ---------- Embedder with fake backend ----------

class _FakeBackend:
    """Deterministic fake so tests never touch the real model."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.encoded_texts: list[str] = []

    def encode(self, texts, batch_size=None, normalize_embeddings=True, show_progress_bar=False):
        self.encoded_texts.extend(texts)
        vectors = []
        for text in texts:
            # Hash-derived deterministic vector, then L2-normalise.
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            v = rng.standard_normal(self.dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-12
            vectors.append(v)
        return np.vstack(vectors) if vectors else np.zeros((0, self.dim), dtype=np.float32)


def test_embedder_returns_normalised_float32():
    embedder = Embedder(backend=_FakeBackend(dim=8))
    out = embedder.encode(["a", "b", "c"])
    assert out.dtype == np.float32
    assert out.shape == (3, 8)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


# ---------- Store round-trip ----------

def _make_post(store, blog_slug, source_id, title="T", body="hello world hello world hello world"):
    raw = RawPost(
        source_post_id=source_id,
        url=f"http://{blog_slug}/{source_id}",
        title=title,
        body_html=f"<p>{body}</p>",
        published_at=datetime(2024, 1, 1),
    )
    return store.upsert_post(blog_slug, raw, normalize(raw))


def test_save_and_iter_embedding_round_trip(tmp_path):
    store = Store(tmp_path / "t.db")
    store.upsert_blog(BlogInfo(slug="b", name="B", url="http://b", platform="rss"))
    pid = _make_post(store, "b", "1")

    vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    store.save_embedding(pid, "test-model", vec)

    loaded = dict(store.iter_embeddings("test-model"))
    assert list(loaded.keys()) == [pid]
    assert np.allclose(loaded[pid], vec)
    store.close()


def test_save_embedding_upserts_on_conflict(tmp_path):
    store = Store(tmp_path / "t.db")
    store.upsert_blog(BlogInfo(slug="b", name="B", url="http://b", platform="rss"))
    pid = _make_post(store, "b", "1")

    store.save_embedding(pid, "m", np.array([1.0, 0.0], dtype=np.float32))
    store.save_embedding(pid, "m", np.array([0.0, 1.0], dtype=np.float32))
    loaded = list(store.iter_embeddings("m"))
    assert len(loaded) == 1
    assert np.allclose(loaded[0][1], [0.0, 1.0])
    store.close()


def test_posts_needing_embedding_skips_already_embedded(tmp_path):
    store = Store(tmp_path / "t.db")
    store.upsert_blog(BlogInfo(slug="b", name="B", url="http://b", platform="rss"))
    pid1 = _make_post(store, "b", "1")
    pid2 = _make_post(store, "b", "2")

    store.save_embedding(pid1, "m", np.array([1.0], dtype=np.float32))
    remaining = store.posts_needing_embedding("m")
    ids = {r["id"] for r in remaining}
    assert ids == {pid2}

    # Different model → both need embedding again under that name.
    assert len(store.posts_needing_embedding("other-model")) == 2
    # --force ignores the extraction state.
    assert len(store.posts_needing_embedding("m", force=True)) == 2
    store.close()


# ---------- similarity edges ----------

def test_build_similarity_edges_top_k_and_floor(tmp_path):
    store = Store(tmp_path / "t.db")
    store.upsert_blog(BlogInfo(slug="b", name="B", url="http://b", platform="rss"))
    ids = [_make_post(store, "b", str(i)) for i in range(4)]

    # Craft four unit vectors in the same 4-D space:
    #  0 and 1 are almost identical (cosine ~0.99)
    #  0 and 2 are moderately aligned (cosine ~0.6)
    #  0 and 3 are orthogonal (cosine 0)
    v = {
        ids[0]: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ids[1]: np.array([0.99, 0.14, 0.0, 0.0], dtype=np.float32),  # ~0.99
        ids[2]: np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32),    # ~0.6
        ids[3]: np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),    # 0
    }
    for pid, vec in v.items():
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        store.save_embedding(pid, "test", vec)

    edges = build_similarity_edges(store, "test", k=10, floor=0.5)
    pairs = {tuple(sorted([e.src_id, e.dst_id])) for e in edges}

    # 0-1 (0.99) and 0-2 (0.6) survive; 0-3 (0) does not; 1-2 (0.6 too) survives.
    assert (ids[0], ids[1]) in pairs or (ids[1], ids[0]) in pairs
    assert (ids[0], ids[2]) in pairs or (ids[2], ids[0]) in pairs
    # No edge to the orthogonal post.
    assert all(ids[3] not in p for p in pairs)
    # All edges are the statistical tier.
    assert {e.source for e in edges} == {"statistical"}
    # No self-edges.
    assert all(e.src_id != e.dst_id for e in edges)
    store.close()


def test_build_similarity_edges_deduplicates_pairs(tmp_path):
    store = Store(tmp_path / "t.db")
    store.upsert_blog(BlogInfo(slug="b", name="B", url="http://b", platform="rss"))
    ids = [_make_post(store, "b", str(i)) for i in range(3)]
    for pid in ids:
        v = np.array([1.0, 0.0], dtype=np.float32)
        store.save_embedding(pid, "test", v)

    edges = build_similarity_edges(store, "test", k=10, floor=0.5)
    # 3 posts, all identical → 3 pairs, undirected, no duplicates.
    assert len(edges) == 3
    pairs = {tuple(sorted([e.src_id, e.dst_id])) for e in edges}
    assert len(pairs) == 3
    store.close()


def test_build_similarity_edges_empty_when_no_vectors(tmp_path):
    store = Store(tmp_path / "t.db")
    assert build_similarity_edges(store, "unused-model", k=10, floor=0.5) == []
    store.close()
