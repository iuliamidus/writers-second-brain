# A Writer's Second Brain

Point it at a set of blogs. Get back a searchable archive and a graph you can
explore, so that fifteen years of scattered writing becomes something you can
actually see the shape of.

Built for a specific problem — a writer with four blogs across two platforms
and two languages, wanting to turn the whole pile into a book — but the
platform-specific parts are confined to one directory, so it generalises.

## Why a graph

Full-text search answers "where did I write about Norway". It cannot answer
"what do I keep coming back to", "which of these posts belong in the same
chapter", or "what connects the music writing to the travel writing". Those
are structural questions, and structure is what a graph is for.

The graph also makes the archive *visible*. A Cypher query that returns a
subgraph draws a picture in Neo4j Browser, and for a writer who has never seen
their own output as a shape, that picture tends to be the moment the tool
justifies itself.

## Architecture

SQLite is the source of truth. Neo4j is a projection you rebuild from it with
one command, which means the graph schema is free to change and the archive
survives even if you never run a graph database at all.

```
blogs ──▶ Source adapters ──▶ normalise ──▶ SQLite ──▶ edge builders ──▶ Neo4j
          (per platform)      (once)        (canonical)                  (disposable)
```

Edges carry provenance and are never blurred together:

| Tier | Where it comes from | Cost | Trust |
|---|---|---|---|
| `structural` | Platform metadata: tags, dates, self-links, series numbering | Free | Fact |
| `statistical` | Embedding similarity between posts | Cheap | Inference |
| `extracted` | An LLM pass over each post | Paid | Inference |

Each tier is stored separately and can be rebuilt or dropped independently, so
you can always ask the graph a question using only the edges you trust.

Build and query the structural tier before reaching for anything cleverer.
Tags, dates and self-links carry more signal than people expect.

## Setup

```bash
git clone <your-repo-url> writers-second-brain
cd writers-second-brain

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
```

Edit `sources.yaml` to list the blogs you want. Then:

```bash
wsb detect https://someblog.wordpress.com   # which adapter handles this?
wsb fetch                                    # pull everything into SQLite
wsb stats                                    # what did we get?
wsb search "Norvegia"                        # diacritic-insensitive full text
```

For the graph:

```bash
docker compose up -d          # Neo4j 5 with the GDS plugin
wsb graph build               # derive structural edges
wsb graph push                # project into Neo4j
```

Open <http://localhost:7474> (`neo4j` / `secondbrain`) and work through
`src/wsb/cypher/queries.cypher`. The queries are ordered by how much of the
pipeline they need; everything in the first three sections runs with
structural edges alone.

## Chatting with the archive

Install the MCP extra and add the server to Claude Desktop's config:

```bash
pip install -e ".[mcp]"
```

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`
(create it if missing) so that `mcpServers` contains:

```json
{
  "mcpServers": {
    "wsb": {
      "command": "/absolute/path/to/writers-second-brain/.venv/bin/wsb",
      "args": ["mcp"],
      "env": {"WSB_DB": "/absolute/path/to/writers-second-brain/second_brain.db"}
    }
  }
}
```

Restart Claude Desktop. The archive is now queryable in plain language:
*"Care sunt scriitorii pe care îi menționez cel mai des?"*, *"Show me every
post about Norway", "Ce teme apar împreună cu singurătatea?"*. Cypher is
also available for structural questions Neo4j can answer.

The server is read-only; it cannot mutate SQLite. Fetching, extraction and
canonicalisation stay CLI-only.

## Design notes

**Diacritics are load-bearing.** Romanian writers mix diacritics and bare
ASCII freely — `carti` in one post, `cărți` in the next — and there are two
incompatible Unicode encodings of ș and ț in circulation (cedilla below,
U+015F, versus the correct comma below, U+0219). Every lexical index folds
all of these together. Skip this and you silently lose half the archive.

**Language detection fails on short posts.** A photo with a two-word caption
is genuinely undetectable, and a corpus like this is full of them. Posts under
twelve words fall back to the blog's configured `default_lang` rather than
guessing.

**Fetching is separate from normalising.** Raw platform JSON is archived to
`data/raw/` and never touched again. Changing a parsing rule should never mean
re-hitting someone's blog.

**Series ordering comes from the writer, not from us.** Titles ending in
`- 2`, `(3)`, `part 2` are the author's own sequencing, so `CONTINUES` edges
are near-certain in a way that inferred edges never are.

## Roadmap

- [x] Source adapters: Blogger, WordPress.com, self-hosted WordPress, RSS
- [x] Normalisation, media extraction, diacritic-folded FTS5
- [x] Structural edges and the Neo4j projection
- [ ] Embeddings (BGE-M3, multilingual) and `SIMILAR_TO` edges
- [ ] LLM entity extraction: people, works, places, themes
- [ ] Quote ledger — flag quoted lyrics and passages with sources, because
      what is fine on a blog is a clearance problem in a printed book
- [ ] Community summaries for corpus-wide questions
- [ ] MCP server, so the archive is queryable from a chat client directly
- [ ] Comment ingestion — reader response is signal the writer already has

## Adding a platform

Write one class in `src/wsb/sources/` satisfying the `Source` protocol
(`describe()` and `fetch()`), register it in `REGISTRY`, and optionally add a
probe to `detect_platform`. Nothing downstream changes.

## Licence

MIT.
