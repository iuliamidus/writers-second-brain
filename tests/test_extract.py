from __future__ import annotations

import json

import pytest

from wsb.db import Store
from wsb.extract import (
    Extractor,
    VALID_KINDS,
    build_user_message,
    parse_response,
    system_prompt,
)


def test_system_prompt_switches_on_language():
    assert "kind must be one of" in system_prompt("en")
    assert "trebuie să fie" in system_prompt("ro")
    assert "kind must be one of" in system_prompt(None)


def test_user_message_uses_romanian_labels_for_ro():
    msg = build_user_message("Titlu", ["etichetă"], "Corp.", "ro")
    assert msg.startswith("TITLU:")
    assert "ETICHETE:" in msg
    assert "TEXT:" in msg


def test_user_message_truncates_long_bodies():
    body = "x" * 10_000
    msg = build_user_message("t", [], body, "en")
    assert msg.count("x") == 3000


def test_parse_response_valid_json():
    payload = json.dumps({
        "entities": [
            {"kind": "person", "name": "Nick Cave", "mention": "Nick", "confidence": 0.9},
            {"kind": "place", "name": "București", "mention": "Bucharest", "confidence": 0.8},
        ]
    })
    out = parse_response(payload)
    assert len(out) == 2
    assert {e.kind for e in out} == {"person", "place"}
    assert out[1].name == "București"


def test_parse_response_drops_below_floor():
    payload = json.dumps({
        "entities": [
            {"kind": "person", "name": "A", "mention": "A", "confidence": 0.9},
            {"kind": "person", "name": "B", "mention": "B", "confidence": 0.3},
        ]
    })
    out = parse_response(payload, floor=0.6)
    assert [e.name for e in out] == ["A"]


def test_parse_response_drops_invalid_kinds():
    payload = json.dumps({
        "entities": [
            {"kind": "genre", "name": "jazz", "mention": "jazz", "confidence": 0.9},
            {"kind": "person", "name": "Leonard Cohen", "mention": "Cohen", "confidence": 0.9},
        ]
    })
    out = parse_response(payload)
    assert [e.kind for e in out] == ["person"]


def test_parse_response_handles_prose_wrapping_json():
    payload = 'Here is the JSON:\n{"entities": [{"kind": "theme", "name": "memory", "mention": "memory", "confidence": 0.7}]}\nHope that helps.'
    out = parse_response(payload)
    assert len(out) == 1
    assert out[0].kind == "theme"


def test_parse_response_handles_garbage():
    assert parse_response("not json at all") == []
    assert parse_response("{malformed") == []
    assert parse_response(json.dumps({"entities": "wrong type"})) == []


def test_parse_response_defaults_missing_mention_to_name():
    payload = json.dumps({
        "entities": [{"kind": "work", "name": "The Boatman's Call", "confidence": 0.9}]
    })
    out = parse_response(payload)
    assert out[0].mention == "The Boatman's Call"


def test_valid_kinds_matches_prompt():
    for kind in VALID_KINDS:
        assert kind in system_prompt("en")
        assert kind in system_prompt("ro")


class _FakeBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _FakeMessage:
    def __init__(self, text: str):
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeMessage(self.reply)


class _FakeClient:
    def __init__(self, reply: str):
        self.messages = _FakeMessages(reply)


def test_extractor_wires_system_and_user_message():
    reply = json.dumps({"entities": [{"kind": "person", "name": "X", "mention": "X", "confidence": 0.9}]})
    client = _FakeClient(reply)
    ext = Extractor(client=client)
    out = ext.extract(title="Title", tags=["t"], body="Body", lang="ro")
    assert out[0].name == "X"
    call = client.messages.calls[0]
    assert "trebuie să fie" in call["system"]
    assert call["messages"][0]["content"].startswith("TITLU:")


def test_store_upsert_entity_merges_diacritic_variants(tmp_path):
    store = Store(tmp_path / "t.db")
    a = store.upsert_entity("place", "București")
    b = store.upsert_entity("place", "Bucuresti")
    c = store.upsert_entity("place", "Bucureşti")  # cedilla ş
    assert a == b == c
    store.close()


def test_store_upsert_entity_keeps_kinds_separate(tmp_path):
    store = Store(tmp_path / "t.db")
    person = store.upsert_entity("person", "Cluj")
    place = store.upsert_entity("place", "Cluj")
    assert person != place
    store.close()


def test_store_upsert_entity_rejects_empty_after_fold(tmp_path):
    store = Store(tmp_path / "t.db")
    with pytest.raises(ValueError):
        store.upsert_entity("theme", "   ")
    store.close()


def test_mark_extracted_makes_posts_needing_extraction_skip(tmp_path):
    from datetime import datetime

    from wsb.models import BlogInfo, RawPost
    from wsb.normalize import normalize

    store = Store(tmp_path / "t.db")
    store.upsert_blog(BlogInfo(slug="b", name="B", url="http://b", platform="rss"))
    raw = RawPost(
        source_post_id="1", url="http://b/1", title="T",
        body_html="<p>hello world hello world hello world hello world</p>",
        published_at=datetime(2024, 1, 1), tags=["tag1"],
    )
    pid = store.upsert_post("b", raw, normalize(raw))

    assert len(store.posts_needing_extraction()) == 1
    store.mark_extracted(pid, raw.content_hash())
    assert store.posts_needing_extraction() == []
    assert len(store.posts_needing_extraction(force=True)) == 1
    store.close()
