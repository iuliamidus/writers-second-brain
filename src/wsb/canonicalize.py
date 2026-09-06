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
from dataclasses import dataclass, field
from datetime import datetime

import yaml

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS = 8192

# Kind-specific guidance. The prompts share a schema but the merge rules
# differ per kind. Anti-examples matter more than positive ones: the model
# will over-merge without explicit "do NOT" cases. All examples come from
# real mistakes seen on this corpus.
KIND_GUIDANCE = {
    "theme": (
        "Merge ONLY if two surface forms describe the same recurring subject:\n"
        "- Cross-language equivalents: 'lockdown' with 'carantină', "
        "'solitude' with 'singurătate', 'memory' with 'amintire'.\n"
        "- Trivial phrasing variants: 'memory' with 'memories'.\n"
        "\n"
        "Do NOT merge related-but-distinct themes. 'grief' stays separate "
        "from 'memory'. 'travel' stays separate from 'solitude'. "
        "Compound themes only merge with their exact cross-language twin."
    ),
    "place": (
        "Merge ONLY when the exact same location is written in different "
        "languages or spellings. Examples of correct merges:\n"
        "- 'Bucharest' with 'București' (English vs Romanian name)\n"
        "- 'Norway' with 'Norvegia' with 'Norge' (three languages, one country)\n"
        "- 'Vienna' with 'Viena', 'Budapest' with 'Budapesta'\n"
        "\n"
        "NEVER merge a place with a region or country that contains it:\n"
        "- 'Bucharest' and 'Romania' are DIFFERENT (city vs country)\n"
        "- 'Paris' and 'Notre-Dame de Paris' are DIFFERENT (city vs monument)\n"
        "- 'Norway' and 'Oslo' are DIFFERENT (country vs capital)\n"
        "- 'USSR' and 'Siberia' are DIFFERENT (country vs region within it)\n"
        "- 'Greece' and 'Plataea' are DIFFERENT (country vs city within it)\n"
        "- 'Czechia' and 'Liberec' are DIFFERENT (country vs city within it)\n"
        "\n"
        "NEVER merge different places even if names sound similar:\n"
        "- 'Belgium' and 'Netherlands' are DIFFERENT countries\n"
        "- 'Munich' and 'Mandalay' are DIFFERENT cities on different continents\n"
        "\n"
        "When in doubt, do NOT merge. Wrong merges silently rewrite geography."
    ),
    "person": (
        "Merge ONLY when clearly the same specific person referred to by "
        "different surface forms:\n"
        "- Full name and unambiguous short form: 'Leonard Cohen' with "
        "'Cohen', 'Nick Cave' with 'Nick Cave'\n"
        "- Same person under a different name: 'Cat Stevens' with "
        "'Yusuf Islam' (same person, name change)\n"
        "- Transliteration variants: 'Kahlil Gibran' with 'Khalil Gibran'\n"
        "\n"
        "NEVER merge based on a common first name alone:\n"
        "- 'David' should NOT be merged with 'David Bowie' or 'David Gilmour' "
        "or 'David Attenborough' — the bare first name is ambiguous.\n"
        "- The same applies to 'Michael', 'John', 'Mihai', 'Ana', 'Maria', "
        "'Iulia', 'Andrei', or any other common first name.\n"
        "\n"
        "NEVER merge different people even if names are similar:\n"
        "- 'Mihai Midus' and 'Iulia Miduș' are DIFFERENT people (family "
        "members share surnames).\n"
        "- Different Popescus, different Ionescus, different Smiths.\n"
        "\n"
        "When in doubt, do NOT merge. False person merges are the worst "
        "kind of error."
    ),
    "work": (
        "Merge ONLY when clearly the same specific work under different "
        "surface forms:\n"
        "- Cross-language titles: '1984' with 'Nineteen Eighty-Four' with "
        "'O mie nouă sute optzeci și patru'\n"
        "- Definite-article variants: 'The Wall' with 'Wall' if clearly "
        "the same album\n"
        "- Subtitle variants: 'Nocturne' with 'Nocturne – Five Stories of "
        "Music and Nightfall' only when clearly the same book\n"
        "\n"
        "NEVER merge different works, even by the same artist:\n"
        "- 'Another Brick in the Wall' (song) is NOT 'The Wall' (album)\n"
        "- 'Pigs' and 'Pigs on the Wing' are DIFFERENT Pink Floyd songs\n"
        "- 'Alternative 3' and 'Alternative 4' are DIFFERENT albums\n"
        "- Two songs on the same album stay separate.\n"
        "- A song and the album containing it stay separate.\n"
        "\n"
        "When titles differ meaningfully but overlap in words, do NOT merge."
    ),
    "org": (
        "Merge cross-language forms and definite-article variants of the "
        "same organisation (e.g. 'Eagles' with 'The Eagles', 'Editura Litera' "
        "with 'Litera' when clearly the same publisher).\n"
        "\n"
        "NEVER merge a parent organisation with a sub-project or brand:\n"
        "- 'Google' is NOT 'Google Art Project' (parent vs project)\n"
        "- 'Google' is NOT 'Google SketchUp' (parent vs product)\n"
        "- 'Virgin Radio' is NOT 'Virgin Radio Classic Rock' (station vs "
        "sister station)\n"
        "\n"
        "NEVER merge different bands, publications or companies that share "
        "a word in their name."
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
    """A proposed merge. `members` are the surface forms that would fold
    into `canonical`; `canonical` itself is never in `members`.
    """

    canonical: str
    members: list[str] = field(default_factory=list)
    approve: bool = False


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
        others = [m for m in deduped if m != canonical]
        out.append(Cluster(canonical=canonical, members=others))
    return out


HEADER_TEMPLATE = """# Entity merge proposals — writers-second-brain
# Generated: {generated}
# Model: {model}
#
# For each cluster below:
#   - Flip `approve: false` to `approve: true` to accept the merge
#   - Leave `approve: false` (or delete the cluster) to reject
#
# The `canonical` name is kept; every entry in `members` is folded into it.
# Original names are preserved in entity_aliases so merges are reversible.
#
# Then run: wsb entities apply-merges <this-file>
"""


def proposals_to_yaml(proposals: dict[str, list[Cluster]], model: str) -> str:
    header = HEADER_TEMPLATE.format(
        generated=datetime.now().isoformat(timespec="seconds"),
        model=model,
    )
    data: dict = {}
    for kind, clusters in proposals.items():
        if not clusters:
            continue
        data[kind] = [
            {
                "canonical": c.canonical,
                "members": list(c.members),
                "approve": False,
            }
            for c in clusters
        ]
    body = yaml.safe_dump(
        data,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    return header + "\n" + body


def load_proposals_yaml(text: str) -> dict[str, list[Cluster]]:
    """Parse a reviewed proposals YAML back into Clusters, keeping the
    reviewer's `approve` flags. Malformed entries are silently dropped so
    a partially-broken file still applies its good rows.
    """
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        return {}
    out: dict[str, list[Cluster]] = {}
    for kind, clusters in data.items():
        if not isinstance(clusters, list):
            continue
        parsed: list[Cluster] = []
        for c in clusters:
            if not isinstance(c, dict):
                continue
            canonical = c.get("canonical")
            members = c.get("members")
            if not isinstance(canonical, str) or not isinstance(members, list):
                continue
            member_strs = [m for m in members if isinstance(m, str) and m != canonical]
            if not member_strs:
                continue
            parsed.append(
                Cluster(
                    canonical=canonical,
                    members=member_strs,
                    approve=bool(c.get("approve", False)),
                )
            )
        out[kind] = parsed
    return out


class Canonicalizer:
    def __init__(self, client=None, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS):
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
