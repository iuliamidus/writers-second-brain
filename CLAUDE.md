# CLAUDE.md

Context for Claude Code working in this repository.

## What this is

A pipeline that ingests a writer's blog archive from multiple platforms and
turns it into a searchable corpus plus a navigable graph. The driving use case
is a writer with four blogs (two Blogger, two WordPress.com), mixed Romanian
and English, roughly 1,000–1,500 posts spanning 2011–present, who wants to
turn the archive into a book.

Two audiences, and they pull in different directions:

1. **The writer.** Non-technical. Interacts through `wsb search` and Neo4j
   Browser. Everything he touches should work without explanation.
2. **The developer.** Learning graph databases and building this as a
   portfolio project. Prefers understanding the mechanism over hiding it.

When those conflict, keep the mechanism visible but the defaults sane.

## Architecture

```
blogs ──▶ Source adapters ──▶ normalise ──▶ SQLite ──▶ edge builders ──▶ Neo4j
          (per platform)      (once)        (canonical)                  (disposable)
```

**SQLite is the source of truth.** Neo4j is a projection, rebuilt with
`wsb graph push --reset`. Never write state to Neo4j that does not exist in
SQLite, and never make the pipeline depend on Neo4j being up. Someone should
be able to use the archive and full-text search with no graph database at all.

**Edges carry provenance and are never blurred.** Every edge has a `source`
of `structural` (platform metadata — a fact), `statistical` (embedding
similarity — an inference) or `extracted` (LLM output — an inference).
`Store.replace_edges` swaps one tier without touching the others. Preserve
this. Being able to query using only trusted edges is a core design property,
not an implementation detail.

## Layout

| Path | Role |
|---|---|
| `src/wsb/models.py` | Dataclasses every stage speaks. Change carefully. |
| `src/wsb/sources/` | The only place platform-specific code belongs. |
| `src/wsb/normalize.py` | HTML → text, media extraction, diacritic folding. |
| `src/wsb/db.py` | SQLite schema and storage. Schema lives in `SCHEMA`. |
| `src/wsb/graph/edges.py` | Edge derivation from stored data. |
| `src/wsb/graph/neo4j_loader.py` | SQLite → Neo4j projection. |
| `src/wsb/cypher/queries.cypher` | Annotated query book, ordered by pipeline stage. |
| `sources.yaml` | Which blogs to ingest. |

## Non-obvious constraints

**Diacritic folding is load-bearing, not a nicety.** Romanian mixes diacritics
and bare ASCII freely — `carti` and `cărți` in the same corpus — and there are
two incompatible Unicode encodings of ș/ț in circulation (U+015F cedilla,
U+0219 comma-below). `normalize.fold()` collapses all of them. Any new lexical
index, entity normaliser or dedupe key must run through it. Getting this wrong
loses half the archive silently, which is the worst kind of bug here.

**Short posts are the common case, not an edge case.** Many posts are one
photo and a caption. Language detection is unreliable below ~12 words and
falls back to the blog's `default_lang`. When embeddings land, do not embed
the two-word body alone — compose title + tags + caption + media context.

**Never re-fetch to fix a parsing bug.** Raw platform JSON is archived to
`data/raw/`. Normalisation must be re-runnable from disk. Be polite to these
servers; they are someone's blog, not an API product.

**Series ordering comes from the author.** `CONTINUES` edges are derived from
titles the writer numbered himself, which makes them near-certain in a way
inferred edges are not. `_series_key` is deliberately conservative — it
rejects short stems so `Life 2` is not treated as a series.

## Conventions

- Python 3.11+, `from __future__ import annotations`, type hints throughout.
- Comments explain *why*, never *what*. If a line needs a comment describing
  what it does, rewrite the line.
- Pipeline stages are separate CLI commands and independently re-runnable.
  Do not collapse them into one `run` command for convenience.
- New dependencies for the embedding and LLM stages go in the relevant
  `optional-dependencies` group, never in the base install. Stage one must
  stay cheap to install.
- `ruff` for lint, line length 100. `pytest` with `pythonpath = ["src"]`.

## Commands

```bash
wsb detect <url>          # which adapter handles this blog
wsb fetch                 # pull everything into SQLite (slow, hits network)
wsb stats                 # corpus summary
wsb search "<query>"      # diacritic-insensitive full text
wsb graph build           # derive edges into SQLite
wsb graph push            # project into Neo4j

docker compose up -d      # Neo4j 5 + GDS on :7474 (neo4j / secondbrain)
pytest -q
```

## Roadmap, in order

1. **Embeddings.** BGE-M3 — multilingual is non-negotiable for this corpus.
   Store vectors in `sqlite-vec`. Emit top-k `SIMILAR_TO` edges as the
   `statistical` tier, pruned with a similarity floor.
2. **Hybrid retrieval.** Reciprocal rank fusion over FTS5 and dense vectors.
3. **LLM extraction.** One pass per post, strict JSON. Entities of kind
   `person`, `work`, `place`, `theme`, `org`. Prompt in the post's own
   language; normalise entity labels through `fold()` so a Romanian post and
   an English one produce the same node.
4. **Quote ledger.** Flag quoted lyrics and passages with their source.
   Acceptable on a blog, a clearance problem in a printed book — this is a
   deliverable for the writer, not a nice-to-have.
5. **Community summaries.** Leiden via GDS, one Claude-written summary per
   community, for corpus-wide questions that no single post answers.
6. **MCP server.** The real interface for the writer: query the archive from
   a chat client in Romanian, no UI to build or learn.

## Working style

Prefer small, reviewable commits over large ones. When adding a pipeline
stage, write the CLI command and the test alongside it. If a change would
break the "SQLite is canonical, Neo4j is disposable" property, stop and
raise it rather than working around it.

Do not add a web frontend, a task queue, or a distributed anything. The corpus
is ~1,500 documents and runs comfortably on a laptop; complexity here is a
cost with no matching benefit.
