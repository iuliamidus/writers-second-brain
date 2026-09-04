"""Blogger / Blogspot adapter.

Blogger's public Atom feed speaks JSON if you ask nicely, needs no API key,
and hands over full post bodies. It caps at 500 entries per request, so we
page with start-index (which is 1-based, not 0-based — a classic trap).
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterator

from wsb.models import BlogInfo, RawPost
from wsb.sources.base import client

PAGE_SIZE = 500


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _text(node: dict | None) -> str:
    return (node or {}).get("$t", "") or ""


class BloggerSource:
    platform = "blogger"

    def __init__(self, url: str, slug: str | None = None, default_lang: str | None = None):
        self.url = url.rstrip("/")
        self.slug = slug or (self.url.split("//")[-1].split(".")[0])
        self.default_lang = default_lang
        self._feed_meta: dict | None = None

    def _feed(self, start_index: int, max_results: int) -> dict:
        with client() as c:
            r = c.get(
                f"{self.url}/feeds/posts/default",
                params={
                    "alt": "json",
                    "max-results": max_results,
                    "start-index": start_index,
                },
            )
            r.raise_for_status()
            return r.json()["feed"]

    def describe(self) -> BlogInfo:
        if self._feed_meta is None:
            self._feed_meta = self._feed(1, 1)
        return BlogInfo(
            slug=self.slug,
            name=_text(self._feed_meta.get("title")) or self.slug,
            url=self.url,
            platform=self.platform,
            default_lang=self.default_lang,
        )

    def fetch(self) -> Iterator[RawPost]:
        start = 1
        while True:
            feed = self._feed(start, PAGE_SIZE)
            entries = feed.get("entry", [])
            if not entries:
                return

            for entry in entries:
                yield self._to_post(entry)

            total = int(_text(feed.get("openSearch$totalResults")) or 0)
            start += len(entries)
            if total and start > total:
                return
            if len(entries) < PAGE_SIZE:
                return

    def _to_post(self, entry: dict) -> RawPost:
        permalink = ""
        for link in entry.get("link", []):
            if link.get("rel") == "alternate":
                permalink = link.get("href", "")
                break

        authors = entry.get("author", [])
        author = _text(authors[0].get("name")) if authors else None

        # Blogger uses <category term="..."> for labels.
        tags = [c["term"] for c in entry.get("category", []) if c.get("term")]

        return RawPost(
            source_post_id=_text(entry.get("id")),
            url=permalink,
            title=_text(entry.get("title")),
            body_html=_text(entry.get("content")) or _text(entry.get("summary")),
            published_at=_parse_dt(_text(entry.get("published"))),
            updated_at=_parse_dt(_text(entry.get("updated"))),
            author=author,
            tags=tags,
            comment_count=int(_text(entry.get("thr$total")) or 0),
            raw=entry,
        )
