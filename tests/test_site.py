from __future__ import annotations

import hashlib
import json
from datetime import datetime

from wsb.db import Store
from wsb.models import BlogInfo, Edge, RawPost
from wsb.normalize import normalize
from wsb.site.build import build_site, build_site_data


def _make_store(tmp_path):
    store = Store(tmp_path / "t.db")
    store.upsert_blog(BlogInfo(slug="a", name="Blog A", url="http://a", platform="rss"))
    ids = []
    for i in range(2):
        raw = RawPost(
            source_post_id=str(i),
            url=f"http://a/{i}",
            title=f"Post {i}",
            body_html=f"<p>this is the body of post {i}, it has some words</p>",
            published_at=datetime(2020 + i, 1, 1),
        )
        ids.append(store.upsert_post("a", raw, normalize(raw)))
    store.replace_edges("statistical", [
        Edge("Post", ids[0], "Post", ids[1], "SIMILAR_TO",
             weight=0.8, source="statistical", confidence=0.8),
    ])
    return store, ids


def test_build_site_data_includes_body_previews(tmp_path):
    store, ids = _make_store(tmp_path)
    data = build_site_data(store)
    assert all("body_preview" in p for p in data["posts"])
    assert any("body of post" in p["body_preview"] for p in data["posts"])
    store.close()


def test_build_site_writes_all_expected_files(tmp_path):
    store, _ = _make_store(tmp_path)
    out = tmp_path / "deploy"
    summary = build_site(store, out, password="hunter2")

    assert (out / "index.html").exists()
    assert (out / "data.json").exists()
    assert (out / "api" / "chat.js").exists()
    assert not (out / "api" / "chat.py").exists()  # replaced by the JS entrypoint
    assert (out / "vercel.json").exists()
    assert summary["posts"] == 2
    assert summary["password_gate"] is True

    html = (out / "index.html").read_text()
    expected_hash = hashlib.sha256(b"hunter2").hexdigest()
    assert expected_hash in html
    assert "__WSB_PASSWORD_HASH__" not in html
    store.close()


def test_build_site_without_password_leaves_hash_empty(tmp_path):
    store, _ = _make_store(tmp_path)
    out = tmp_path / "deploy"
    build_site(store, out, password=None)
    html = (out / "index.html").read_text()
    # Placeholder replaced with empty string — client treats as public.
    assert 'PASSWORD_HASH = ""' in html
    store.close()


def test_data_json_no_secret_leakage(tmp_path):
    store, _ = _make_store(tmp_path)
    out = tmp_path / "deploy"
    build_site(store, out, password="hunter2")
    data_text = (out / "data.json").read_text()
    assert "ANTHROPIC" not in data_text
    assert "sk-ant" not in data_text
    # And the password hash never lands in the client data payload.
    assert hashlib.sha256(b"hunter2").hexdigest() not in data_text
    store.close()


def test_api_chat_stub_has_no_hardcoded_key(tmp_path):
    store, _ = _make_store(tmp_path)
    out = tmp_path / "deploy"
    build_site(store, out, password="hunter2")
    api_text = (out / "api" / "chat.js").read_text()
    assert "sk-ant" not in api_text
    assert "process.env.ANTHROPIC_API_KEY" in api_text
    store.close()


def test_data_json_is_valid_json(tmp_path):
    store, _ = _make_store(tmp_path)
    out = tmp_path / "deploy"
    build_site(store, out)
    parsed = json.loads((out / "data.json").read_text())
    assert "posts" in parsed and "blogs" in parsed and "edges" in parsed
    store.close()
