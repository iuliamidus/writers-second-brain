"""LLM-driven entity canonicalisation.

The extraction pass emits entities in whatever language the post was written
in, so "lockdown" and "carantină" end up as separate theme nodes. This module
sends the full list of entity names of a given kind to Claude and asks it to
propose a merge mapping: which surface forms describe the same thing, and
which of them should be the canonical name.

Only *cross-form* duplicates are surfaced — singletons don't appear in the
output. The canonical name must be one of the members; the model cannot
invent new labels here. All merges pass through `Store.merge_entities`, which
preserves the original names in `entity_aliases` so nothing is silently lost.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Kind-specific guidance. The prompts share a schema but the merge rules
# differ: for people you almost never merge; for themes you often do.
KIND_GUIDANCE = {
    "theme": (
        "Themes are recurring subjects. Merge cross-language equivalents "
        "(e.g. 'lockdown' with 'carantină', 'solitude' with 'singurătate') "
        "and merge minor phrasing variants of the same concept "
        "(e.g. 'memory' with 'memories', 'călătorie solitară' with 'solo travel'). "
        "Do NOT merge related-but-distinct themes (e.g. 'grief' and 'memory' "
        "stay separate)."
    ),
    "place": (
        "Places should merge across language and spelling (e.g. 'Bucharest' "
        "with 'București', 'Norway' with 'Norvegia', 'Norge'). "
        "Do NOT merge different places even if names are similar."
    ),
    "person": (
        "Merge only when clearly the same person referred to by different "
        "surface forms (e.g. 'Cohen' with 'Leonard Cohen', 'Nick' with "
        "'Nick Cave'). When in doubt, do NOT merge — false merges are worse "
        "than duplicates for people."
    ),
    "work": (
        "Merge translated titles and abbreviations of the same work "
        "(e.g. '1984' with 'Nineteen Eighty-Four'). Do NOT merge different "
        "works by the same author."
    ),
    "org": (
        "Merge cross-language forms and short-forms of the same organisation. "
        "Do NOT merge different bands, publications or companies."
    ),
}

SYSTEM_TEMPLATE = """You canonicalise a list of {kind} entities extracted from
a bilingual (Romanian + English) personal blog archive.

{guidance}

Return strict JSON matching this schema and nothing else:
{{"clusters": [{{"canonical": "...", "members": ["...", "...", "..."]}}]}}

Rules:
- Only include clusters with 2 or more members. Never emit singletons.
- The `canonical` value MUST be one of the strings in `members`.
- Members are exact strings from the input list. Do not paraphrase or invent.
- Prefer the Romanian spelling with correct diacritics as canonical when the
  entity has a Romanian name; otherwise prefer the most complete form.
- Each input string may appear in at most one cluster.
- If nothing needs merging, return {{"clusters": []}}."""


@dataclass
class Cluster:
    canonical: str
    members: list[str]


def system_prompt(kind: str) -> str:
    guidance = KIND_GUIDANCE.get(kind, "Merge only clear duplicates of the same underlying entity.")
    return SYSTEM_TEMPLATE.format(kind=kind, guidance=guidance)


def build_user_message(names: list[str]) -> str:
    numbered = "\n".join(f"- {n}" for n in names)
    return f"Entities to review:\n{numbered}"


def parse_response(text: str, valid_names: set[str]) -> list[Cluster]:
    """Parse and validate the model's cluster JSON.

    Enforces:
    - `canonical` is a member of `members`
    - every member appears in `valid_names` (rejects hallucinated strings)
    - members are unique within a cluster
    - each name is claimed by at most one cluster (first cluster wins)
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return []
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []

    raw = payload.get("clusters") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []

    out: list[Cluster] = []
    claimed: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        canonical = item.get("canonical")
        members = item.get("members")
        if not isinstance(canonical, str) or not isinstance(members, list):
            continue
        deduped: list[str] = []
        for m in members:
            if not isinstance(m, str):
                continue
            if m not in valid_names or m in claimed or m in deduped:
                continue
            deduped.append(m)
        if canonical not in deduped or len(deduped) < 2:
            continue
        for m in deduped:
            claimed.add(m)
        out.append(Cluster(canonical=canonical, members=deduped))
    return out


class Canonicalizer:
    def __init__(self, client=None, model: str = DEFAULT_MODEL, max_tokens: int = 4096):
        if client is None:
            from anthropic import Anthropic

            client = Anthropic()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def propose(self, kind: str, names: list[str]) -> list[Cluster]:
        if not names:
            return []
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt(kind),
            messages=[{"role": "user", "content": build_user_message(names)}],
        )
        text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
        return parse_response(text, set(names))
