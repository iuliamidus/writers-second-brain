"""Handler-level tests for the MCP server.

We exercise the plain Python functions rather than going through the MCP
transport — the transport is thin glue over these handlers, and testing it
would just be testing the SDK.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from wsb.db import Store
from wsb.mcp_server import (
    corpus_stats,
    entity_neighbors,
    get_post,
    handle_call,
    list_entities,
    posts_about,
    search_posts,
)
from wsb.models import BlogInfo, RawPost
from wsb.normalize import normalize


@pytest.fixture
def db(tmp_path):
    """A small corpus with two posts across two blogs, mixed languages,
    and a handful of entities including diacritic variants.
    """
    path = tmp_path / "t.db"
    store = Store(path)
    store.upsert_blog(BlogInfo(slug="ro", name="Ro Blog", url="http://ro", platform="rss", default_lang="ro"))
    store.upsert_blog(BlogInfo(slug="en", name="En Blog", url="http://en", platform="rss", default_lang="en"))

    raw1 = RawPost(
        source_post_id="1",
        url="http://ro/1",
        title="Amintiri din București",
        body_html="<p>M-am plimbat prin București împreună cu Nichita Stănescu, "
                  "un mare poet român, admirând orașul frumos.</p>",
        published_at=datetime(2024, 3, 1),
        tags=["București", "poezie"],
    )
    pid1 = store.upsert_post("ro", raw1, normalize(raw1, default_lang="ro"))

    raw2 = RawPost(
        source_post_id="2",
        url="http://en/2",
        title="A walk in Bucharest",
        body_html="<p>Walking through Bucharest, thinking of Nichita Stanescu, a great Romanian poet.</p>",
        published_at=datetime(2024, 3, 15),
        tags=["Bucharest", "poetry"],
    )
    pid2 = store.upsert_post("en", raw2, normalize(raw2, default_lang="en"))

    # Both posts mention the same city and person (with diacritic variants).
    bucuresti = store.upsert_entity("place", "București")
    poet = store.upsert_entity("person", "Nichita Stănescu")
    store.replace_post_entities(pid1, [(bucuresti, "București", 0.95), (poet, "Nichita Stănescu", 0.9)])
    # English post uses non-diacritic spellings; fold() collapses to same entities.
    bucharest = store.upsert_entity("place", "Bucharest")  # different normalized due to English word
    store.replace_post_entities(pid2, [(bucharest, "Bucharest", 0.9), (poet, "Nichita Stanescu", 0.9)])
    store.conn.commit()

    store.close()
    return str(path), pid1, pid2


def test_search_posts_diacritic_insensitive(db):
    path, pid1, _ = db
    # Bare-ASCII query hits the diacritic-having body ("București").
    results = search_posts(path, "Bucuresti")
    ids = {r["id"] for r in results}
    assert pid1 in ids
    # Sanity check the reverse: query with diacritics still matches.
    results2 = search_posts(path, "București")
    assert pid1 in {r["id"] for r in results2}


def test_search_posts_returns_snippet_and_score(db):
    path, _, _ = db
    results = search_posts(path, "poet")
    assert results
    assert "snippet" in results[0]
    assert "score" in results[0]


def test_search_posts_respects_limit(db):
    path, _, _ = db
    results = search_posts(path, "București", limit=1)
    assert len(results) <= 1


def test_get_post_returns_body_tags_entities(db):
    path, pid1, _ = db
    post = get_post(path, pid1)
    assert post is not None
    assert post["id"] == pid1
    assert "București" in post["body_text"]
    assert set(post["tags"]) >= {"București", "poezie"}
    kinds = {e["kind"] for e in post["entities"]}
    assert "place" in kinds
    assert "person" in kinds


def test_get_post_missing_returns_none(db):
    path, _, _ = db
    assert get_post(path, "nope:404") is None


def test_list_entities_all_kinds(db):
    path, _, _ = db
    entities = list_entities(path, kind="all", min_mentions=1)
    kinds = {e["kind"] for e in entities}
    assert kinds == {"place", "person"}


def test_list_entities_filters_by_kind(db):
    path, _, _ = db
    persons = list_entities(path, kind="person", min_mentions=1)
    assert all(e["kind"] == "person" for e in persons)


def test_list_entities_rejects_bad_kind(db):
    path, _, _ = db
    with pytest.raises(ValueError):
        list_entities(path, kind="unicorn")


def test_list_entities_respects_min_mentions(db):
    path, _, _ = db
    # The person is mentioned in both posts, so it survives min_mentions=2.
    # The city has two normalized forms and each is mentioned once.
    top = list_entities(path, kind="person", min_mentions=2)
    assert len(top) == 1


def test_posts_about_diacritic_insensitive(db):
    path, pid1, pid2 = db
    # Query with bare ASCII, should hit the diacritic-having entity.
    results = posts_about(path, "Bucuresti", kind="place")
    ids = {r["id"] for r in results}
    # Only the Romanian post — the English one is under a different `normalized`
    # key ("bucharest" vs "bucuresti"), which reflects the actual cross-language
    # merge story (an issue canonicalize solves separately).
    assert pid1 in ids


def test_posts_about_unknown_entity_returns_empty(db):
    path, _, _ = db
    assert posts_about(path, "Someone Nobody Knows") == []


def test_entity_neighbors_excludes_self(db):
    path, _, _ = db
    neighbors = entity_neighbors(path, "Nichita Stănescu")
    names = {n["name"] for n in neighbors}
    assert "Nichita Stănescu" not in names
    # The poet co-occurs with both spellings of the city.
    assert names == {"București", "Bucharest"}


def test_corpus_stats_smoke(db):
    path, _, _ = db
    stats = corpus_stats(path)
    assert stats["posts"] == 2
    assert stats["blogs"] == 2
    assert stats["entities"] >= 3


def test_handle_call_dispatches_by_name(db):
    path, pid1, _ = db
    result = asyncio.run(handle_call("get_post", {"post_id": pid1}, path))
    assert result["id"] == pid1


def test_handle_call_unknown_tool_raises(db):
    path, _, _ = db
    with pytest.raises(ValueError):
        asyncio.run(handle_call("does_not_exist", {}, path))
