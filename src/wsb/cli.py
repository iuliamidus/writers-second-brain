"""Command line interface.

Stages are separate commands on purpose. Fetching is slow and polite;
normalising is cheap and re-runnable; the graph is disposable. Keeping them
apart means you never re-hit someone's blog because you changed a regex.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from wsb.db import Store
from wsb.graph.edges import build_entity_edges, build_structural_edges
from wsb.normalize import normalize
from wsb.sources import build_source, detect_platform

app = typer.Typer(add_completion=False, help="A Writer's Second Brain")
graph_app = typer.Typer(help="Graph projection commands")
entities_app = typer.Typer(help="Entity maintenance commands")
app.add_typer(graph_app, name="graph")
app.add_typer(entities_app, name="entities")

console = Console()
DEFAULT_DB = "second_brain.db"
DEFAULT_CONFIG = "sources.yaml"


def _load_config(path: str) -> list[dict]:
    config = yaml.safe_load(Path(path).read_text())
    return config.get("sources", [])


@app.command()
def detect(url: str):
    """Work out which adapter handles a blog URL."""
    console.print(f"[bold]{url}[/bold] → [cyan]{detect_platform(url)}[/cyan]")


@app.command()
def fetch(
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    db: str = typer.Option(DEFAULT_DB, "--db"),
    only: str = typer.Option(None, "--only", help="Fetch a single blog by slug"),
    archive_dir: str = typer.Option("data/raw", "--archive"),
):
    """Pull every post from every configured blog into SQLite.

    Raw platform JSON is also written to disk. Blogs die and platforms change
    their APIs; that archive is the one artefact worth keeping regardless.
    """
    store = Store(db)
    archive = Path(archive_dir)
    archive.mkdir(parents=True, exist_ok=True)

    for entry in _load_config(config):
        if only and entry.get("slug") != only:
            continue

        source = build_source(
            entry["url"],
            platform=entry.get("platform"),
            slug=entry.get("slug"),
            default_lang=entry.get("default_lang"),
        )
        info = source.describe()
        store.upsert_blog(info)
        console.print(f"\n[bold cyan]{info.name}[/bold cyan]  ({info.platform})")

        count = 0
        raw_dump = []
        with console.status("fetching…") as status:
            for raw in source.fetch():
                norm = normalize(raw, default_lang=info.default_lang)
                store.upsert_post(info.slug, raw, norm)
                raw_dump.append(raw.raw)
                count += 1
                status.update(f"fetching… {count} posts")

        (archive / f"{info.slug}.json").write_text(
            json.dumps(raw_dump, default=str, ensure_ascii=False, indent=2)
        )
        console.print(f"  [green]{count}[/green] posts stored")

    store.close()


@app.command()
def stats(db: str = typer.Option(DEFAULT_DB, "--db")):
    """Summarise what is in the archive."""
    store = Store(db)
    s = store.stats()

    table = Table(title="Corpus")
    for col in ("Blog", "Posts", "First", "Last"):
        table.add_column(col)
    for row in s["by_blog"]:
        table.add_row(
            row["name"] or row["slug"],
            str(row["posts"]),
            (row["first"] or "")[:10],
            (row["last"] or "")[:10],
        )
    console.print(table)

    langs = ", ".join(f"{r['lang'] or 'unknown'}: {r['n']}" for r in s["by_lang"])
    console.print(f"\nLanguages  {langs}")
    console.print(
        f"Tags {s['tags']}   Media {s['media']}   "
        f"Entities {s['entities']}   Edges {s['edges']}"
    )
    store.close()


@app.command()
def search(
    query: str,
    db: str = typer.Option(DEFAULT_DB, "--db"),
    limit: int = typer.Option(15, "--limit", "-n"),
):
    """Full-text search across every blog, diacritic-insensitive.

    'carti' and 'cărți' return the same results, which is the entire reason
    this exists rather than using the blogs' own search boxes.
    """
    store = Store(db)
    results = store.search(query, limit)
    if not results:
        console.print("[yellow]No matches.[/yellow]")
        store.close()
        return

    for row in results:
        date = (row["published_at"] or "")[:10]
        console.print(
            f"\n[bold]{row['title'] or '(untitled)'}[/bold] "
            f"[dim]{date} · {row['blog_slug']} · {row['lang'] or '?'}[/dim]"
        )
        console.print(f"  {row['snip']}")
        console.print(f"  [blue]{row['url']}[/blue]")
    store.close()


@app.command()
def extract(
    db: str = typer.Option(DEFAULT_DB, "--db"),
    blog: str = typer.Option(None, "--blog", help="Restrict to one blog slug"),
    limit: int = typer.Option(None, "--limit", "-n", help="Cap number of posts"),
    force: bool = typer.Option(False, "--force", help="Re-extract already-processed posts"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print what would be sent, do not call the API"),
    floor: float = typer.Option(0.6, "--floor", help="Confidence floor for kept entities"),
    model: str = typer.Option(None, "--model", help="Anthropic model id override"),
):
    """LLM entity extraction pass — populates the `extracted` tier.

    Idempotent: skips posts whose content_hash already has an extraction row,
    so re-running only touches new or edited posts. Use --force to redo.
    """
    from wsb.extract import Extractor, build_user_message, system_prompt

    store = Store(db)
    posts = store.posts_needing_extraction(blog=blog, limit=limit, force=force)
    if not posts:
        console.print("[yellow]Nothing to extract.[/yellow]")
        store.close()
        return

    console.print(f"[cyan]{len(posts)}[/cyan] posts queued for extraction")

    if dry_run:
        for row in posts[:3]:
            tags = (row["tags"] or "").split("||") if row["tags"] else []
            console.print(f"\n[bold]{row['title']}[/bold] [dim]{row['blog_slug']} · {row['lang']}[/dim]")
            console.print("[dim]--- system ---[/dim]")
            console.print(system_prompt(row["lang"]))
            console.print("[dim]--- user ---[/dim]")
            console.print(build_user_message(row["title"] or "", tags, row["body_text"] or "", row["lang"] or "en"))
        console.print(f"\n[dim](showing 3 of {len(posts)}; dry-run — no API call made)[/dim]")
        store.close()
        return

    extractor = Extractor(model=model) if model else Extractor()

    kept = 0
    skipped = 0
    with console.status("extracting…") as status:
        for i, row in enumerate(posts, 1):
            tags = (row["tags"] or "").split("||") if row["tags"] else []
            try:
                entities = extractor.extract(
                    title=row["title"] or "",
                    tags=tags,
                    body=row["body_text"] or "",
                    lang=row["lang"],
                    floor=floor,
                )
            except Exception as exc:
                console.print(f"[red]error on {row['id']}: {exc}[/red]")
                skipped += 1
                continue

            mentions = []
            for e in entities:
                try:
                    entity_id = store.upsert_entity(e.kind, e.name)
                except ValueError:
                    continue
                mentions.append((entity_id, e.mention, e.confidence))

            store.replace_post_entities(row["id"], mentions)
            store.mark_extracted(row["id"], row["content_hash"])
            kept += len(mentions)
            status.update(f"extracting… {i}/{len(posts)} posts · {kept} entities kept")

    console.print(f"[green]{kept}[/green] entity mentions stored across {len(posts) - skipped} posts")
    if skipped:
        console.print(f"[yellow]{skipped}[/yellow] posts skipped due to API errors")
    console.print("Next: [dim]wsb graph build && wsb graph push[/dim]")
    store.close()


@entities_app.command("propose-merges")
def entities_propose_merges(
    output: Path = typer.Argument(..., help="YAML file to write proposals to"),
    db: str = typer.Option(DEFAULT_DB, "--db"),
    kind: str = typer.Option("all", "--kind", help="person, work, place, theme, org, or 'all'"),
    model: str = typer.Option(None, "--model"),
):
    """Step 1 of the merge workflow: ask the LLM for merge proposals, write
    them to a YAML file for human review.

    LLM output is non-deterministic and imperfect. The workflow is deliberately
    two-step: propose → review → apply. Never apply merges without reading
    them, because false merges silently rewrite the graph.
    """
    from wsb.canonicalize import Canonicalizer, proposals_to_yaml

    kinds = ["person", "work", "place", "theme", "org"] if kind == "all" else [kind]
    store = Store(db)
    canon = Canonicalizer(model=model) if model else Canonicalizer()

    proposals: dict[str, list] = {}
    for k in kinds:
        rows = store.entities_by_kind(k)
        if not rows:
            console.print(f"[dim]{k}: no entities[/dim]")
            continue
        names = [r["name"] for r in rows]
        console.print(f"[cyan]{k}[/cyan]  {len(names)} entities → asking model to cluster…")

        clusters = canon.propose(k, names)
        proposals[k] = clusters
        console.print(f"  [green]{len(clusters)}[/green] clusters proposed")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(proposals_to_yaml(proposals, model=canon.model))
    store.close()

    total = sum(len(v) for v in proposals.values())
    console.print(f"\n[green]{total}[/green] proposals written to [blue]{output}[/blue]")
    console.print("Now open the file, flip [dim]approve: false[/dim] to [dim]approve: true[/dim] on the ones you want,")
    console.print(f"then run: [dim]wsb entities apply-merges {output}[/dim]")


@entities_app.command("apply-merges")
def entities_apply_merges(
    input_file: Path = typer.Argument(..., help="Reviewed YAML from propose-merges"),
    db: str = typer.Option(DEFAULT_DB, "--db"),
):
    """Step 2 of the merge workflow: apply only the clusters you marked
    `approve: true` in the reviewed YAML file.

    Members are folded into the canonical entity; original surface forms are
    preserved in `entity_aliases` so the merge is reversible.
    """
    from wsb.canonicalize import load_proposals_yaml

    proposals = load_proposals_yaml(input_file.read_text())
    store = Store(db)

    total_merged = 0
    total_rejected = 0
    total_missing = 0
    for kind, clusters in proposals.items():
        approved = [c for c in clusters if c.approve]
        rejected = len(clusters) - len(approved)
        total_rejected += rejected
        if not approved:
            console.print(f"[dim]{kind}: nothing approved (of {len(clusters)})[/dim]")
            continue

        console.print(f"\n[bold cyan]{kind}[/bold cyan] applying {len(approved)} approved merges")
        for cluster in approved:
            canonical_id = store.upsert_entity(kind, cluster.canonical) if False else None
            # Look up existing entities; do NOT create if missing.
            canonical_row = store.conn.execute(
                "SELECT id FROM entities WHERE kind = ? AND normalized = ?",
                (kind, _fold_lookup(cluster.canonical)),
            ).fetchone()
            member_ids: list[int] = []
            missing: list[str] = []
            for m in cluster.members:
                row = store.conn.execute(
                    "SELECT id FROM entities WHERE kind = ? AND normalized = ?",
                    (kind, _fold_lookup(m)),
                ).fetchone()
                if row is None:
                    missing.append(m)
                elif canonical_row is None or row["id"] != canonical_row["id"]:
                    member_ids.append(row["id"])

            if canonical_row is None:
                console.print(f"  [yellow]skip[/yellow] canonical {cluster.canonical!r} no longer exists")
                total_missing += 1
                continue
            if missing:
                console.print(f"  [dim]  {len(missing)} member(s) missing: {', '.join(missing)}[/dim]")
                total_missing += len(missing)
            if not member_ids:
                console.print(f"  [dim]  no members to merge into {cluster.canonical}[/dim]")
                continue

            store.merge_entities(canonical_row["id"], member_ids)
            total_merged += len(member_ids)
            console.print(f"  [green]{cluster.canonical}[/green] ← {', '.join(cluster.members)}")

    store.close()
    console.print(
        f"\n[green]{total_merged}[/green] entities merged  ·  "
        f"[dim]{total_rejected} clusters rejected · {total_missing} names missing[/dim]"
    )
    console.print("Rebuild the graph: [dim]wsb graph build && wsb graph push[/dim]")


def _fold_lookup(name: str) -> str:
    """Same normalisation the entity table uses. Kept local to avoid a wider
    import surface in this command file."""
    from wsb.normalize import fold
    return fold(name).strip()


@app.command()
def mcp(db: str = typer.Option(None, "--db", help="DB path; defaults to $WSB_DB or ./second_brain.db")):
    """Run the MCP server over stdio for chat-client integration.

    Point Claude Desktop (or any MCP client) at this command. All output is
    the MCP protocol on stdout — do not print anything else here.

    The DB path resolution is explicit: --db flag wins, else $WSB_DB, else
    the working directory. Claude Desktop launches this from an arbitrary
    cwd, so relative paths do not work there — use $WSB_DB in the config.
    """
    import asyncio
    import os

    from wsb.mcp_server import serve

    db_path = db or os.getenv("WSB_DB") or DEFAULT_DB
    asyncio.run(serve(db_path))


@graph_app.command("build")
def graph_build(db: str = typer.Option(DEFAULT_DB, "--db")):
    """Derive structural edges from platform metadata."""
    store = Store(db)
    structural = build_structural_edges(store)
    store.replace_edges("structural", structural)
    console.print(f"[green]{len(structural)}[/green] structural edges")

    extracted = build_entity_edges(store)
    if extracted:
        store.replace_edges("extracted", extracted)
        console.print(f"[green]{len(extracted)}[/green] extracted edges")

    by_rel: dict[str, int] = {}
    for edge in structural + extracted:
        by_rel[edge.rel] = by_rel.get(edge.rel, 0) + 1
    for rel, n in sorted(by_rel.items(), key=lambda kv: -kv[1]):
        console.print(f"  {rel:<14} {n}")
    store.close()


@graph_app.command("push")
def graph_push(
    db: str = typer.Option(DEFAULT_DB, "--db"),
    reset: bool = typer.Option(True, "--reset/--no-reset"),
):
    """Project SQLite into Neo4j. Safe to re-run; nothing lives only here."""
    from wsb.graph.neo4j_loader import Neo4jLoader

    store = Store(db)
    loader = Neo4jLoader()
    counts = loader.push(store, reset=reset)
    loader.close()
    store.close()

    console.print("[green]Pushed to Neo4j[/green]")
    for key, value in counts.items():
        console.print(f"  {key:<10} {value}")
    console.print("\nOpen [blue]http://localhost:7474[/blue] and try:")
    console.print("  [dim]MATCH (p:Post)-[:PUBLISHED_ON]->(b:Blog) RETURN b.name, count(p)[/dim]")


if __name__ == "__main__":
    app()
