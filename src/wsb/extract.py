"""LLM entity extraction — the `extracted` provenance tier.

One call per post, strict JSON out. Bilingual: the prompt is chosen from the
post's detected language so the model reads and thinks in the same language
the writer wrote in. The entity `name` field is the canonical form the model
chose; SQLite merges by `fold(name)` so diacritic variants collapse into one
node regardless of prompt language.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

VALID_KINDS = {"person", "work", "place", "theme", "org"}
MAX_BODY_CHARS = 3000
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_CONFIDENCE_FLOOR = 0.6


@dataclass
class Extracted:
    kind: str
    name: str
    mention: str
    confidence: float


SYSTEM_EN = """You extract named entities from a personal blog post.

Return strict JSON matching this schema and nothing else:
{"entities": [{"kind": "...", "name": "...", "mention": "...", "confidence": 0.0}]}

Rules:
- kind must be one of: person, work, place, theme, org
- person: real named people (writers, musicians, friends). Not pronouns.
- work: named creative works (books, albums, songs, films). Not genres.
- place: named geographic places (cities, countries, venues). Not directions.
- org: named organisations (bands, publications, companies). A band that
  released an album is an org; the album is a work.
- theme: a recurring subject the post is actually about, in 1-3 words.
  Prefer nouns the writer would recognise ("solitude", "memory", "exile")
  over generic ones ("life", "things"). Emit at most 3 themes per post.
- name is the canonical form. When the entity has a native Romanian name,
  use the Romanian spelling with correct diacritics (București, not
  Bucharest; Nichita Stănescu, not Nichita Stanescu).
- mention is the exact surface form as it appears in the post.
- confidence is your own certainty from 0.0 to 1.0.
- Return an empty entities list rather than guessing.
- Do not extract anything from tags alone; the body must support it."""

SYSTEM_RO = """Extragi entităţi numite dintr-un articol de blog personal.

Returnează strict JSON conform acestei scheme şi nimic altceva:
{"entities": [{"kind": "...", "name": "...", "mention": "...", "confidence": 0.0}]}

Reguli:
- kind trebuie să fie unul dintre: person, work, place, theme, org
- person: persoane reale numite (scriitori, muzicieni, prieteni). Nu pronume.
- work: opere creative numite (cărţi, albume, cântece, filme). Nu genuri.
- place: locuri geografice numite (oraşe, ţări, săli). Nu direcţii.
- org: organizaţii numite (trupe, publicaţii, companii). O trupă care a
  lansat un album este org; albumul este work.
- theme: un subiect recurent despre care este articolul, în 1-3 cuvinte.
  Preferă substantive pe care autorul le-ar recunoaşte ("singurătate",
  "amintire", "exil") în locul unora generice ("viaţă", "lucruri").
  Cel mult 3 teme per articol.
- name este forma canonică. Foloseşte grafia românească cu diacritice
  corecte (București, Nichita Stănescu, Cluj-Napoca).
- mention este forma exactă aşa cum apare în articol.
- confidence este certitudinea ta între 0.0 şi 1.0.
- Returnează o listă goală în loc să ghiceşti.
- Nu extrage nimic doar din etichete; trebuie să fie susţinut de text."""


def build_user_message(title: str, tags: list[str], body: str, lang: str) -> str:
    """Assemble the post payload the model reads.

    Short-post safety: we always include title and tags because bodies can be
    a single caption. Body is truncated to keep per-call cost predictable;
    chunking long essays is a future problem.
    """
    body = (body or "")[:MAX_BODY_CHARS]
    tag_str = ", ".join(tags) if tags else "(none)"
    if lang == "ro":
        return (
            f"TITLU: {title or '(fără titlu)'}\n"
            f"ETICHETE: {tag_str}\n\n"
            f"TEXT:\n{body}"
        )
    return (
        f"TITLE: {title or '(untitled)'}\n"
        f"TAGS: {tag_str}\n\n"
        f"BODY:\n{body}"
    )


def system_prompt(lang: str | None) -> str:
    return SYSTEM_RO if lang == "ro" else SYSTEM_EN


def parse_response(text: str, floor: float = DEFAULT_CONFIDENCE_FLOOR) -> list[Extracted]:
    """Parse the model's JSON reply into validated Extracted rows.

    Anything malformed or off-schema is dropped rather than raising: the batch
    should keep moving even when one post yields garbage. Callers can log the
    drop count if they want visibility.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []

    raw = payload.get("entities") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []

    out: list[Extracted] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        name = item.get("name")
        if kind not in VALID_KINDS or not isinstance(name, str) or not name.strip():
            continue
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if confidence < floor:
            continue
        mention = item.get("mention") if isinstance(item.get("mention"), str) else name
        out.append(
            Extracted(
                kind=kind,
                name=name.strip(),
                mention=(mention or name).strip(),
                confidence=confidence,
            )
        )
    return out


class Extractor:
    """Thin wrapper around the Anthropic client so tests can substitute it."""

    def __init__(self, client=None, model: str = DEFAULT_MODEL, max_tokens: int = 1024):
        if client is None:
            from anthropic import Anthropic

            client = Anthropic()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def extract(
        self,
        title: str,
        tags: list[str],
        body: str,
        lang: str | None,
        floor: float = DEFAULT_CONFIDENCE_FLOOR,
    ) -> list[Extracted]:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt(lang),
            messages=[{"role": "user", "content": build_user_message(title, tags, body, lang or "en")}],
        )
        text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
        return parse_response(text, floor=floor)
