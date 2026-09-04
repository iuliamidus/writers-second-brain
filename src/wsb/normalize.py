"""Turn platform HTML into clean text plus structured media.

The important piece here is `fold`. Romanian writers routinely mix diacritics
and bare ASCII in the same corpus — "carti" in one post, "cărți" in the next —
and there are two incompatible Unicode encodings of ș/ț in circulation
(cedilla below, U+015F, and the correct comma below, U+0219). Any search that
does not fold all of these together will silently miss half the archive.
"""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from markdownify import markdownify

from wsb.models import MediaItem, NormalizedPost, RawPost

YOUTUBE_RE = re.compile(
    r"(?:youtube\.com/(?:embed/|watch\?v=)|youtu\.be/)([A-Za-z0-9_-]{11})"
)

# Legacy cedilla forms that NFKD alone leaves looking different from the
# comma-below forms. Mapped explicitly so both spellings collapse to one.
_ROMANIAN_FIXUPS = str.maketrans({
    "ş": "s", "Ş": "S", "ţ": "t", "Ţ": "T",
    "ș": "s", "Ș": "S", "ț": "t", "Ț": "T",
})


def fold(text: str) -> str:
    """Lowercase and strip every diacritic, for lexical matching only.

    Never store this as the display text — it is a search key, not content.
    """
    text = text.translate(_ROMANIAN_FIXUPS)
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.lower()


def detect_lang(text: str, fallback: str | None = None) -> str | None:
    """Best-effort language detection.

    Short posts — a photo and a caption — are genuinely undetectable, so we
    require a minimum length and otherwise defer to the blog's default.
    """
    if len(text.split()) < 12:
        return fallback
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(text)
    except Exception:
        return fallback


def _extract_media(soup: BeautifulSoup, post_url: str) -> tuple[list[MediaItem], list[str]]:
    media: list[MediaItem] = []
    links: list[str] = []
    seen_video: set[str] = set()
    post_host = urlparse(post_url).netloc

    for img in soup.find_all("img"):
        src = img.get("src")
        if src:
            media.append(MediaItem("image", src, img.get("alt") or img.get("title")))

    for tag in soup.find_all(["iframe", "embed", "a"]):
        target = tag.get("src") or tag.get("href") or ""
        match = YOUTUBE_RE.search(target)
        if match and match.group(1) not in seen_video:
            seen_video.add(match.group(1))
            media.append(MediaItem("video", f"https://youtu.be/{match.group(1)}"))

    for a in soup.find_all("a"):
        href = a.get("href") or ""
        if not href.startswith("http"):
            continue
        if YOUTUBE_RE.search(href):
            continue
        # Keep self-links: they are how the writer connected their own posts,
        # and they become LINKS_TO edges later.
        links.append(href)

    return media, links


def normalize(raw: RawPost, default_lang: str | None = None) -> NormalizedPost:
    soup = BeautifulSoup(raw.body_html or "", "html.parser")
    media, links = _extract_media(soup, raw.url)

    body_md = markdownify(raw.body_html or "", heading_style="ATX").strip()
    body_text = soup.get_text(separator=" ", strip=True)
    body_text = re.sub(r"\s+", " ", body_text)

    return NormalizedPost(
        source_post_id=raw.source_post_id,
        url=raw.url,
        title=raw.title,
        body_md=body_md,
        body_text=body_text,
        search_text=fold(f"{raw.title} {body_text} {' '.join(raw.tags)}"),
        lang=detect_lang(body_text, default_lang),
        word_count=len(body_text.split()),
        media=media,
        outbound_links=links,
    )
