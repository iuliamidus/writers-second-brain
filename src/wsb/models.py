"""Platform-neutral data model.

Everything downstream of a Source adapter speaks these types only. If you are
adding support for a new blogging platform, your job is to produce RawPost
objects and nothing else.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class BlogInfo:
    """Identity of a single blog within the corpus."""

    slug: str
    name: str
    url: str
    platform: str
    default_lang: str | None = None


@dataclass
class RawPost:
    """A post exactly as the source platform gave it to us.

    `body_html` is untouched platform HTML. Normalisation happens later, so
    that re-running normalisation never requires re-fetching.
    """

    source_post_id: str
    url: str
    title: str
    body_html: str
    published_at: datetime | None
    updated_at: datetime | None = None
    author: str | None = None
    tags: list[str] = field(default_factory=list)
    comment_count: int = 0
    raw: dict = field(default_factory=dict)

    def content_hash(self) -> str:
        """Stable hash used to skip re-processing unchanged posts."""
        payload = f"{self.title}\x00{self.body_html}\x00{','.join(sorted(self.tags))}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class MediaItem:
    kind: str  # image | video | link
    url: str
    caption: str | None = None


@dataclass
class NormalizedPost:
    """A RawPost after HTML stripping, language detection and media extraction."""

    source_post_id: str
    url: str
    title: str
    body_md: str
    body_text: str
    search_text: str  # diacritic-folded, lowercased
    lang: str | None
    word_count: int
    media: list[MediaItem] = field(default_factory=list)
    outbound_links: list[str] = field(default_factory=list)


@dataclass
class Edge:
    """One relationship in the graph.

    `source` records provenance and is never blurred: 'structural' edges are
    facts from the platform, 'statistical' come from embeddings, 'extracted'
    come from an LLM. Keeping them separable means you can always ask the
    graph a question using only the edges you actually trust.
    """

    src_type: str
    src_id: str
    dst_type: str
    dst_id: str
    rel: str
    weight: float = 1.0
    source: str = "structural"  # structural | statistical | extracted
    confidence: float = 1.0
