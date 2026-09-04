"""Source registry.

`build_source` is the only function the rest of the codebase should call.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterator

import feedparser

from wsb.models import BlogInfo, RawPost
from wsb.sources.base import Source, detect_platform
from wsb.sources.blogger import BloggerSource
from wsb.sources.wordpress import WordPressComSource, WordPressSelfHostedSource


class RssSource:
    """Last-resort adapter.

    Most feeds only expose recent posts and often only excerpts, so this gets
    you a partial archive at best. It exists so the tool never hard-fails on
    an unknown platform.
    """

    platform = "rss"

    def __init__(self, url: str, slug: str | None = None, default_lang: str | None = None):
        self.url = url.rstrip("/")
        self.slug = slug or (self.url.split("//")[-1].split(".")[0])
        self.default_lang = default_lang
        self._parsed = None

    def _feed(self):
        if self._parsed is None:
            self._parsed = feedparser.parse(self.url)
            if not self._parsed.entries:
                for suffix in ("/feed", "/rss", "/atom.xml", "/index.xml"):
                    self._parsed = feedparser.parse(self.url + suffix)
                    if self._parsed.entries:
                        break
        return self._parsed

    def describe(self) -> BlogInfo:
        feed = self._feed().feed
        return BlogInfo(
            slug=self.slug,
            name=feed.get("title", self.slug),
            url=self.url,
            platform=self.platform,
            default_lang=self.default_lang or feed.get("language"),
        )

    def fetch(self) -> Iterator[RawPost]:
        for entry in self._feed().entries:
            body = ""
            if entry.get("content"):
                body = entry["content"][0].get("value", "")
            body = body or entry.get("summary", "")

            published = None
            if entry.get("published_parsed"):
                published = datetime(*entry["published_parsed"][:6])

            yield RawPost(
                source_post_id=entry.get("id") or entry.get("link", ""),
                url=entry.get("link", ""),
                title=entry.get("title", ""),
                body_html=body,
                published_at=published,
                author=entry.get("author"),
                tags=[t.get("term") for t in entry.get("tags", []) if t.get("term")],
                raw=dict(entry),
            )


REGISTRY = {
    "blogger": BloggerSource,
    "wordpress_com": WordPressComSource,
    "wordpress_self": WordPressSelfHostedSource,
    "rss": RssSource,
}


def build_source(
    url: str,
    platform: str | None = None,
    slug: str | None = None,
    default_lang: str | None = None,
) -> Source:
    """Return the right adapter for a blog URL, probing the site if needed."""
    platform = platform or detect_platform(url)
    if platform not in REGISTRY:
        raise ValueError(f"Unknown platform {platform!r}. Known: {', '.join(REGISTRY)}")
    return REGISTRY[platform](url, slug=slug, default_lang=default_lang)


__all__ = ["build_source", "detect_platform", "REGISTRY", "Source"]
