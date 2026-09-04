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
app.add_typer(graph_app, name="graph")

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
