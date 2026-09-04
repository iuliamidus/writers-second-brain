"""WordPress adapters — hosted (.wordpress.com) and self-hosted.

The hosted API needs no auth for public sites. The self-hosted one uses the
standard /wp-json/wp/v2 REST API, which is enabled by default since WP 4.7.
"""

from __future__ import annotations

import html
from datetime import datetime
from typing import Iterator

from wsb.models import BlogInfo, RawPost
from wsb.sources.base import client

WPCOM_API = "https://public-api.wordpress.com/rest/v1.1/sites"
PAGE_SIZE = 100


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class WordPressComSource:
    platform = "wordpress_com"

    def __init__(self, url: str, slug: str | None = None, default_lang: str | None = None):
        self.url = url.rstrip("/")
        self.site = self.url.split("//")[-1].strip("/")
        self.slug = slug or self.site.split(".")[0]
        self.default_lang = default_lang

    def describe(self) -> BlogInfo:
        with client() as c:
            r = c.get(f"{WPCOM_API}/{self.site}")
            r.raise_for_status()
            data = r.json()
        return BlogInfo(
            slug=self.slug,
            name=data.get("name") or self.slug,
            url=self.url,
            platform=self.platform,
            default_lang=self.default_lang or (data.get("lang") or None),
        )

    def fetch(self) -> Iterator[RawPost]:
        offset = 0
        with client() as c:
            while True:
                r = c.get(
                    f"{WPCOM_API}/{self.site}/posts/",
                    params={"number": PAGE_SIZE, "offset": offset, "status": "publish"},
                )
                r.raise_for_status()
                posts = r.json().get("posts", [])
                if not posts:
                    return

                for post in posts:
                    yield self._to_post(post)

                offset += len(posts)
                if len(posts) < PAGE_SIZE:
                    return

    def _to_post(self, post: dict) -> RawPost:
        # Categories and tags are dicts keyed by name; both are useful labels
        # and we deliberately flatten them together.
        tags = list(post.get("categories", {}).keys()) + list(post.get("tags", {}).keys())

        return RawPost(
            source_post_id=str(post.get("ID")),
            url=post.get("URL", ""),
            title=html.unescape(post.get("title", "")),
            body_html=post.get("content", ""),
            published_at=_parse_dt(post.get("date")),
            updated_at=_parse_dt(post.get("modified")),
            author=(post.get("author") or {}).get("name"),
            tags=tags,
            comment_count=int(post.get("discussion", {}).get("comment_count") or 0),
            raw=post,
        )


class WordPressSelfHostedSource:
    platform = "wordpress_self"

    def __init__(self, url: str, slug: str | None = None, default_lang: str | None = None):
        self.url = url.rstrip("/")
        self.slug = slug or (self.url.split("//")[-1].split(".")[0])
        self.default_lang = default_lang

    def describe(self) -> BlogInfo:
        name = self.slug
        try:
            with client() as c:
                r = c.get(f"{self.url}/wp-json")
                if r.status_code == 200:
                    name = r.json().get("name") or name
        except Exception:
            pass
        return BlogInfo(self.slug, name, self.url, self.platform, self.default_lang)

    def fetch(self) -> Iterator[RawPost]:
        page = 1
        with client() as c:
            while True:
                r = c.get(
                    f"{self.url}/wp-json/wp/v2/posts",
                    params={"per_page": PAGE_SIZE, "page": page, "_embed": "1"},
                )
                if r.status_code == 400:  # past the last page
                    return
                r.raise_for_status()
                posts = r.json()
                if not posts:
                    return

                for post in posts:
                    yield self._to_post(post)

                page += 1
                if len(posts) < PAGE_SIZE:
                    return

    def _to_post(self, post: dict) -> RawPost:
        embedded = post.get("_embedded", {})
        terms = embedded.get("wp:term", [])
        tags = [t["name"] for group in terms for t in group if t.get("name")]
        authors = embedded.get("author", [])

        return RawPost(
            source_post_id=str(post.get("id")),
            url=post.get("link", ""),
            title=html.unescape((post.get("title") or {}).get("rendered", "")),
            body_html=(post.get("content") or {}).get("rendered", ""),
            published_at=_parse_dt(post.get("date_gmt")),
            updated_at=_parse_dt(post.get("modified_gmt")),
            author=authors[0].get("name") if authors else None,
            tags=tags,
            comment_count=0,
            raw=post,
        )
