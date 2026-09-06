from __future__ import annotations

import json

from wsb.canonicalize import Canonicalizer, Cluster, parse_response, system_prompt
from wsb.db import Store


def test_system_prompt_specialises_by_kind():
    assert "themes" in system_prompt("theme").lower()
    assert "places" in system_prompt("place").lower()
    assert "same person" in system_prompt("person").lower()


def test_parse_response_valid():
    payload = json.dumps({
        "clusters": [
            {"canonical": "carantină", "members": ["carantină", "lockdown", "quarantine"]},
            {"canonical": "București", "members": ["București", "Bucharest"]},
        ]
    })
    out = parse_response(payload, valid_names={"carantină", "lockdown", "quarantine", "București", "Bucharest"})
    assert len(out) == 2
    assert out[0].canonical == "carantină"
    assert set(out[0].members) == {"carantină", "lockdown", "quarantine"}


def test_parse_response_rejects_hallucinated_members():
    payload = json.dumps({
        "clusters": [
            {"canonical": "carantină", "members": ["carantină", "invented-name"]}
        ]
    })
    out = parse_response(payload, valid_names={"carantină", "lockdown"})
    assert out == []  # only 1 valid member left → cluster dropped


def test_parse_response_rejects_canonical_not_in_members():
    payload = json.dumps({
        "clusters": [
            {"canonical": "quarantine-canonical", "members": ["carantină", "lockdown"]}
        ]
    })
    out = parse_response(payload, valid_names={"carantină", "lockdown"})
    assert out == []


def test_parse_response_deduplicates_across_clusters():
    payload = json.dumps({
        "clusters": [
            {"canonical": "a", "members": ["a", "b"]},
            {"canonical": "b", "members": ["b", "c"]},  # b already claimed
        ]
    })
    out = parse_response(payload, valid_names={"a", "b", "c"})
    assert len(out) == 1
    assert set(out[0].members) == {"a", "b"}


def test_parse_response_handles_garbage():
    assert parse_response("nope", valid_names={"a"}) == []
    assert parse_response(json.dumps({"clusters": "wrong"}), valid_names={"a"}) == []


def test_parse_response_ignores_singletons():
    payload = json.dumps({"clusters": [{"canonical": "a", "members": ["a"]}]})
    assert parse_response(payload, valid_names={"a"}) == []


class _FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeMessage:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]


class _FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    class _M:
        def __init__(self, outer):
            self.outer = outer

        def create(self, **kwargs):
            self.outer.calls.append(kwargs)
            return _FakeMessage(self.outer.reply)

    @property
    def messages(self):
        return self._M(self)


def test_canonicalizer_returns_parsed_clusters():
    reply = json.dumps({
        "clusters": [{"canonical": "a", "members": ["a", "b"]}]
    })
    client = _FakeClient(reply)
    canon = Canonicalizer(client=client)
    out = canon.propose("theme", ["a", "b", "c"])
    assert out == [Cluster(canonical="a", members=["a", "b"])]


def test_merge_entities_redirects_and_records_aliases(tmp_path):
    from datetime import datetime

    from wsb.models import BlogInfo, RawPost
    from wsb.normalize import normalize

    store = Store(tmp_path / "t.db")
    store.upsert_blog(BlogInfo(slug="b", name="B", url="http://b", platform="rss"))
    raw = RawPost(
        source_post_id="1", url="http://b/1", title="T",
        body_html="<p>hello world hello world hello world hello world</p>",
        published_at=datetime(2024, 1, 1),
    )
    pid = store.upsert_post("b", raw, normalize(raw))

    canonical = store.upsert_entity("theme", "carantină")
    member_a = store.upsert_entity("theme", "lockdown")
    member_b = store.upsert_entity("theme", "quarantine")
    store.replace_post_entities(pid, [
        (member_a, "lockdown", 0.8),
        (member_b, "quarantine", 0.7),
    ])

    redirected = store.merge_entities(canonical, [member_a, member_b])
    assert redirected == 1  # first insert; second is same post so replaces

    remaining_entities = {r["name"] for r in store.conn.execute("SELECT name FROM entities")}
    assert remaining_entities == {"carantină"}

    aliases = {r["alias"] for r in store.conn.execute("SELECT alias FROM entity_aliases")}
    assert aliases == {"lockdown", "quarantine"}

    mentions = list(store.conn.execute(
        "SELECT entity_id, mention FROM post_entities WHERE post_id = ?", (pid,)
    ))
    assert len(mentions) == 1
    assert mentions[0]["entity_id"] == canonical

    store.close()


def test_merge_entities_keeps_highest_confidence(tmp_path):
    from datetime import datetime

    from wsb.models import BlogInfo, RawPost
    from wsb.normalize import normalize

    store = Store(tmp_path / "t.db")
    store.upsert_blog(BlogInfo(slug="b", name="B", url="http://b", platform="rss"))
    raw = RawPost(
        source_post_id="1", url="http://b/1", title="T",
        body_html="<p>hello world hello world hello world hello world</p>",
        published_at=datetime(2024, 1, 1),
    )
    pid = store.upsert_post("b", raw, normalize(raw))

    canonical = store.upsert_entity("theme", "solitude")
    member = store.upsert_entity("theme", "singurătate")
    store.replace_post_entities(pid, [
        (canonical, "solitude", 0.6),
        (member, "singurătate", 0.95),
    ])

    store.merge_entities(canonical, [member])

    mentions = list(store.conn.execute(
        "SELECT confidence, mention FROM post_entities WHERE post_id = ?", (pid,)
    ))
    assert len(mentions) == 1
    assert mentions[0]["confidence"] == 0.95
    assert mentions[0]["mention"] == "singurătate"
    store.close()
