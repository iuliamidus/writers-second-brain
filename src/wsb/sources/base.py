"""The Source interface — the single seam that makes this tool general.

Everything platform-specific lives in an adapter. Adding Substack or Medium
means writing one class here, and nothing downstream changes.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

import httpx

from wsb.models import BlogInfo, RawPost

USER_AGENT = "writers-second-brain/0.1 (personal archive tool)"
TIMEOUT = httpx.Timeout(30.0, connect=10.0)


@runtime_checkable
class Source(Protocol):
    """A blog we can pull posts from."""

    platform: str

    def describe(self) -> BlogInfo:
        """Identity of this blog, without fetching all its posts."""
        ...

    def fetch(self) -> Iterator[RawPost]:
        """Yield every post, oldest or newest first — order does not matter."""
        ...


def client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT,
        follow_redirects=True,
    )


def detect_platform(url: str) -> str:
    """Probe a URL to work out which adapter to use.

    Cheap heuristics first, then an actual request. This is what lets a user
    paste a bare URL and have the tool figure the rest out.
    """
    host = httpx.URL(url).host or ""

    if host.endswith(".blogspot.com") or ".blogspot." in host:
        return "blogger"
    if host.endswith(".wordpress.com"):
        return "wordpress_com"

    with client() as c:
        # Self-hosted WordPress exposes the REST API at /wp-json/wp/v2.
        try:
            r = c.get(f"{url.rstrip('/')}/wp-json/wp/v2/posts", params={"per_page": 1})
            if r.status_code == 200 and isinstance(r.json(), list):
                return "wordpress_self"
        except Exception:
            pass

        # Blogger on a custom domain still serves the Atom feed.
        try:
            r = c.get(f"{url.rstrip('/')}/feeds/posts/default", params={"alt": "json", "max-results": 1})
            if r.status_code == 200 and "feed" in r.json():
                return "blogger"
        except Exception:
            pass

    return "rss"
