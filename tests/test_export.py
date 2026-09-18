from __future__ import annotations

import json
from datetime import datetime

from wsb.db import Store
from wsb.graph.export import build_graph_json, render_html
from wsb.models import BlogInfo, Edge, RawPost
from wsb.normalize import normalize


def _make_store(tmp_path):
    store = Store(tmp_path / "t.db")
    store.upsert_blog(BlogInfo(slug="a", name="Blog A", url="http://a", platform="rss"))
    store.upsert_blog(BlogInfo(slug="b", name="Blog B", url="http://b", platform="rss"))

    ids = []
    for i, slug in enumerate(["a", "a", "b"]):
        raw = RawPost(
            source_post_id=str(i),
            url=f"http://{slug}/{i}",
            title=f"Post {i}",
            body_html=f"<p>body {i}</p>",
            published_at=datetime(2020 + i, 6, 1),
            tags=[f"tag{i}"],
        )
        ids.append(store.upsert_post(slug, raw, normalize(raw)))

    e_id = store.upsert_entity("person", "Someone")
    store.replace_post_entities(ids[0], [(e_id, "Someone", 0.9)])

    store.replace_edges("statistical", [
        Edge("Post", ids[0], "Post", ids[1], "SIMILAR_TO",
             weight=0.85, source="statistical", confidence=0.85),
        Edge("Post", ids[0], "Post", ids[2], "SIMILAR_TO",
             weight=0.4, source="statistical", confidence=0.4),  # below default floor
    ])
    return store, ids


def test_build_graph_json_shape(tmp_path):
    store, ids = _make_store(tmp_path)
    graph = build_graph_json(store, min_similarity=0.5)

    assert {b["slug"] for b in graph["blogs"]} == {"a", "b"}
    assert {p["id"] for p in graph["posts"]} == set(ids)
    assert all("title" in p and "blog" in p and "year" in p for p in graph["posts"])

    # min_similarity filters out the 0.4 edge.
    weights = [e["weight"] for e in graph["edges"]["similar"]]
    assert len(weights) == 1
    assert weights[0] == 0.85

    assert graph["meta"]["year_min"] <= graph["meta"]["year_max"]
    store.close()


def test_build_graph_json_min_similarity_prunes(tmp_path):
    store, _ = _make_store(tmp_path)
    strict = build_graph_json(store, min_similarity=0.9)
    assert strict["edges"]["similar"] == []
    store.close()


def test_render_html_embeds_data_and_escapes_script_end(tmp_path):
    store, _ = _make_store(tmp_path)
    graph = build_graph_json(store)
    html = render_html(graph)
    assert "cytoscape" in html
    assert "__WSB_GRAPH_JSON__" not in html
    # Round-trip: extract the JSON, parse it, verify structure.
    marker = "const GRAPH = "
    start = html.index(marker) + len(marker)
    end = html.index(";", start)
    payload = html[start:end]
    # No unescaped </script> inside the embedded JSON could end the tag early.
    assert "</script>" not in payload
    parsed = json.loads(payload.replace("<\\/", "</"))
    assert "posts" in parsed and "edges" in parsed
    store.close()


def test_render_html_no_leaked_secrets(tmp_path):
    store, _ = _make_store(tmp_path)
    graph = build_graph_json(store)
    html = render_html(graph)
    # No env or filesystem paths should leak; the template must not reference
    # absolute paths from the machine that built it.
    assert "/Users/" not in html
    assert "sk-ant" not in html
    assert "ANTHROPIC" not in html
    store.close()
