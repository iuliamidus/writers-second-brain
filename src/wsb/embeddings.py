"""Dense multilingual embeddings — the `statistical` provenance tier.

Every post gets one vector under a named model. Composition matters more than
model choice here: a two-word caption gives a garbage vector by itself, so
each embedding text stitches title + tags + body + media captions together
before being sent to the encoder. That is not a nicety; it is the difference
between finding related short posts and burying them.

BGE-M3 is the default because the corpus is bilingual (Romanian + English)
and BGE-M3 is one of the few open models that treats both first-class. The
Embedder wraps the loader so tests can inject a fake without pulling a
2GB model download into CI.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_BODY_CHARS = 2000  # cap so long essays do not dominate a batch


def compose_text(
    title: str | None,
    tags: list[str] | None,
    body: str | None,
    media_captions: list[str] | None,
    max_body_chars: int = DEFAULT_BODY_CHARS,
) -> str:
    """Assemble the string to embed. Everything is optional; short posts
    where only a caption exists still get useful signal via title + tags."""
    parts: list[str] = []
    if title and title.strip():
        parts.append(title.strip())
    if tags:
        cleaned_tags = [t.strip() for t in tags if t and t.strip()]
        if cleaned_tags:
            parts.append("Tags: " + ", ".join(cleaned_tags))
    if body and body.strip():
        parts.append(body.strip()[:max_body_chars])
    if media_captions:
        cleaned = [c.strip() for c in media_captions if c and c.strip()]
        if cleaned:
            parts.append("Captions: " + " · ".join(cleaned))
    return "\n\n".join(parts)


def _split_pipe(field: str | None) -> list[str]:
    if not field:
        return []
    return [p for p in field.split("||") if p.strip()]


def compose_from_row(row, max_body_chars: int = DEFAULT_BODY_CHARS) -> str:
    """Convenience wrapper for the SQL rows returned by
    Store.posts_needing_embedding — unpacks the GROUP_CONCAT columns."""
    return compose_text(
        title=row["title"],
        tags=_split_pipe(row["tags"]),
        body=row["body_text"],
        media_captions=_split_pipe(row["captions"]),
        max_body_chars=max_body_chars,
    )


@dataclass
class EmbedResult:
    post_id: str
    vector: object  # numpy.ndarray, kept opaque to avoid importing numpy at type-check


class Embedder:
    """Thin wrapper around sentence-transformers so tests can substitute a fake."""

    def __init__(self, model_name: str = DEFAULT_MODEL, backend=None, device: str | None = None):
        self.model_name = model_name
        if backend is None:
            from sentence_transformers import SentenceTransformer

            backend = SentenceTransformer(model_name, device=device)
        self.backend = backend

    def encode(self, texts: list[str], batch_size: int = 8):
        """Encode a batch; returns an (n, dim) float32 numpy array,
        L2-normalised so cosine similarity is a plain dot product."""
        import numpy as np

        vectors = self.backend.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
