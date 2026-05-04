"""AgentBase CLI — Command-line interface for the context database."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from agentbase_core.engine import AgentBaseEngine
from agentbase_core.models.config import AgentBaseConfig
from agentbase_core.models import ContextScope, ContextType, MemoryCategory

app = typer.Typer(
    name="agentbase",
    help="AgentBase — Context Database for AI Agents",
    no_args_is_help=True,
)
session_app = typer.Typer(help="Session management commands.")
graph_app = typer.Typer(help="Knowledge graph commands.")
app.add_typer(session_app, name="session")
app.add_typer(graph_app, name="graph")

console = Console()


def _default_db_path() -> Path:
    """Return the default database path."""
    return Path.cwd() / "agentbase.db"


def _make_engine(db_path: Path | None = None) -> AgentBaseEngine:
    """Create and return an initialized engine (sync, for CLI use)."""
    path = db_path or _default_db_path()
    config = AgentBaseConfig(data_dir=path.parent, db_filename=path.name)
    engine = AgentBaseEngine(config=config)
    asyncio.run(engine.initialize())
    return engine


def _close_engine(engine: AgentBaseEngine) -> None:
    """Close the engine cleanly."""
    asyncio.run(engine.close())


# ── init ──────────────────────────────────────────────────────────────────


@app.command()
def init(
    data_dir: Optional[str] = typer.Option(
        None, "--data-dir", help="Data directory path"
    ),
) -> None:
    """Initialize an AgentBase database."""
    path = Path(data_dir) if data_dir else _default_db_path()
    config = AgentBaseConfig(data_dir=path.parent, db_filename=path.name)
    engine = AgentBaseEngine(config=config)
    asyncio.run(engine.initialize())
    asyncio.run(engine.close())
    console.print(f"[green]✓[/green] AgentBase initialized: {path}")


# ── add ───────────────────────────────────────────────────────────────────


@app.command("add")
def add_entry(
    content: str = typer.Argument(..., help="Content text"),
    type: str = typer.Option("memory", "--type", "-t", help="Entry type: memory|resource|skill"),
    category: Optional[str] = typer.Option(
        None, "--category", help="Memory category (for type=memory)"
    ),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
    scope: str = typer.Option("global", "--scope", help="Scope: global|agent|project|session"),
    owner: Optional[str] = typer.Option(None, "--owner", help="Owner agent/project ID"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database file path"),
) -> None:
    """Add a context entry."""
    engine = _make_engine(Path(db_path) if db_path else None)
    try:
        context_type = ContextType(type)
        context_scope = ContextScope(scope)
        memory_cat = MemoryCategory(category) if category and context_type == ContextType.MEMORY else None
        tag_list = [t.strip() for t in tags.split(",")] if tags else None

        if context_type == ContextType.MEMORY:
            entry = asyncio.run(
                engine.add_memory(
                    content=content,
                    category=memory_cat,
                    tags=tag_list,
                    scope=context_scope,
                    owner_id=owner,
                )
            )
        elif context_type == ContextType.RESOURCE:
            entry = asyncio.run(
                engine.add_resource(
                    content=content,
                    tags=tag_list,
                    scope=context_scope,
                )
            )
        elif context_type == ContextType.SKILL:
            entry = asyncio.run(
                engine.add_skill(
                    tool_name=content,
                    tags=tag_list,
                )
            )
        else:
            raise typer.BadParameter(f"Unsupported type: {type}")

        console.print(f"[green]✓[/green] Added entry: {entry.id}")
    finally:
        _close_engine(engine)


# ── search / find ─────────────────────────────────────────────────────────


@app.command("find")
def find(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by type"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database file path"),
) -> None:
    """Search for context entries."""
    engine = _make_engine(Path(db_path) if db_path else None)
    try:
        context_type = ContextType(type) if type else None
        results = asyncio.run(engine.find(query, top_k=top_k, context_type=context_type))

        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return

        table = Table(title="Search Results")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta")
        table.add_column("Score", style="green")
        table.add_column("Content", max_width=60)

        for r in results:
            table.add_row(
                str(r.entry.id),
                r.entry.context_type.value,
                f"{r.score:.3f}",
                (r.entry.l2_full or "")[:60],
            )
        console.print(table)
    finally:
        _close_engine(engine)


@app.command("search")
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database file path"),
) -> None:
    """Search for context entries (alias for find)."""
    find(query=query, top_k=top_k, type=None, db_path=db_path)


# ── get ───────────────────────────────────────────────────────────────────


@app.command("get")
def get(
    entry_id: str = typer.Argument(..., help="Entry ID"),
    level: Optional[str] = typer.Option(None, "--level", help="Content level: l0|l1|l2"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database file path"),
) -> None:
    """Get a context entry by ID."""
    engine = _make_engine(Path(db_path) if db_path else None)
    try:
        entry = asyncio.run(engine.get(entry_id))
        if entry is None:
            console.print(f"[red]✗[/red] Entry not found: {entry_id}")
            return

        content = entry.get_content(level) if level and hasattr(entry, "get_content") else entry.l2_full
        console.print(f"[cyan]ID:[/cyan]       {entry.id}")
        console.print(f"[cyan]Type:[/cyan]     {entry.context_type.value}")
        console.print(f"[cyan]Scope:[/cyan]    {entry.scope.value}")
        console.print(f"[cyan]Status:[/cyan]   {entry.status.value}")
        console.print(f"[cyan]Tags:[/cyan]     {', '.join(entry.tags or [])}")
        console.print(f"[cyan]Content:[/cyan]\n{content}")
    finally:
        _close_engine(engine)


# ── list ──────────────────────────────────────────────────────────────────


@app.command("list")
def list_entries(
    scope: Optional[str] = typer.Option(None, "--scope", help="Filter by scope"),
    type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by type"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max entries to show"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database file path"),
) -> None:
    """List context entries."""
    engine = _make_engine(Path(db_path) if db_path else None)
    try:
        context_scope = ContextScope(scope) if scope else None
        context_type = ContextType(type) if type else None
        entries = asyncio.run(
            engine.list_entries(scope=context_scope, context_type=context_type, limit=limit)
        )

        if not entries:
            console.print("[yellow]No entries found.[/yellow]")
            return

        table = Table(title="Context Entries")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta")
        table.add_column("Scope", style="blue")
        table.add_column("Status", style="green")
        table.add_column("Content", max_width=50)

        for e in entries:
            table.add_row(
                str(e.id),
                e.context_type.value,
                e.scope.value,
                e.status.value,
                (e.l2_full or "")[:50],
            )
        console.print(table)
    finally:
        _close_engine(engine)


# ── delete / purge ────────────────────────────────────────────────────────


@app.command("delete")
def delete(
    entry_id: str = typer.Argument(..., help="Entry ID"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database file path"),
) -> None:
    """Soft-delete a context entry."""
    engine = _make_engine(Path(db_path) if db_path else None)
    try:
        asyncio.run(engine.delete(entry_id))
        console.print(f"[green]✓[/green] Deleted: {entry_id}")
    finally:
        _close_engine(engine)


@app.command("purge")
def purge(
    entry_id: str = typer.Argument(..., help="Entry ID"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database file path"),
) -> None:
    """Permanently delete a context entry."""
    engine = _make_engine(Path(db_path) if db_path else None)
    try:
        asyncio.run(engine.purge(entry_id))
        console.print(f"[green]✓[/green] Purged: {entry_id}")
    finally:
        _close_engine(engine)


@app.command("cleanup")
def cleanup(
    db_path: Optional[str] = typer.Option(None, "--db", help="Database file path"),
) -> None:
    """Clean up soft-deleted entries."""
    engine = _make_engine(Path(db_path) if db_path else None)
    try:
        count = asyncio.run(engine.cleanup())
        console.print(f"[green]✓[/green] Cleaned up {count} entries")
    finally:
        _close_engine(engine)


# ── session commands ──────────────────────────────────────────────────────


@session_app.command("create")
def session_create(
    agent_id: Optional[str] = typer.Option(None, "--agent-id", help="Agent ID"),
    project: Optional[str] = typer.Option(None, "--project", help="Project ID"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database file path"),
) -> None:
    """Create a new conversation session."""
    engine = _make_engine(Path(db_path) if db_path else None)
    try:
        session = asyncio.run(engine.create_session(agent_id=agent_id, project=project))
        console.print(f"[green]✓[/green] Session created: {session.id}")
    finally:
        _close_engine(engine)


@session_app.command("add-message")
def session_add_message(
    session_id: str = typer.Argument(..., help="Session ID"),
    role: str = typer.Option(..., "--role", help="Message role: user|assistant|system"),
    content: str = typer.Option(..., "--content", "-c", help="Message content"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database file path"),
) -> None:
    """Add a message to a session."""
    engine = _make_engine(Path(db_path) if db_path else None)
    try:
        msg = asyncio.run(engine.add_message(session_id=session_id, role=role, content=content))
        console.print(f"[green]✓[/green] Message added: {msg.id}")
    finally:
        _close_engine(engine)


@session_app.command("show")
def session_show(
    session_id: str = typer.Argument(..., help="Session ID"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database file path"),
) -> None:
    """Show session details with messages."""
    engine = _make_engine(Path(db_path) if db_path else None)
    try:
        session = asyncio.run(engine.get_session(session_id))
        if session is None:
            console.print(f"[red]✗[/red] Session not found: {session_id}")
            return
        console.print(f"[cyan]Session ID:[/cyan] {session.id}")
        console.print(f"[cyan]Agent ID:[/cyan]   {session.agent_id or 'N/A'}")
        console.print(f"[cyan]Status:[/cyan]     {session.status}")
        if session.messages:
            for msg in session.messages:
                console.print(f"  [{msg.role}] {msg.content[:100]}")
    finally:
        _close_engine(engine)


@session_app.command("commit")
def session_commit(
    session_id: str = typer.Argument(..., help="Session ID"),
    mode: str = typer.Option("full", "--mode", help="Commit mode: full|summary"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database file path"),
) -> None:
    """Commit a session and extract memories."""
    engine = _make_engine(Path(db_path) if db_path else None)
    try:
        memories = asyncio.run(engine.commit_session(session_id=session_id, mode=mode))
        console.print(f"[green]✓[/green] Committed session, extracted {len(memories)} memories")
    finally:
        _close_engine(engine)


# ── graph commands ────────────────────────────────────────────────────────


@graph_app.command("query")
def graph_query(
    name: str = typer.Argument(..., help="Entity name to query"),
    depth: int = typer.Option(2, "--depth", "-d", help="Traversal depth"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database file path"),
) -> None:
    """Query the knowledge graph."""
    engine = _make_engine(Path(db_path) if db_path else None)
    try:
        results = asyncio.run(engine.graph_traversal(name, max_depth=depth))
        if not results:
            console.print(f"[yellow]No graph results for: {name}[/yellow]")
            return

        table = Table(title=f"Graph: {name}")
        table.add_column("Entity", style="cyan")
        table.add_column("Relation", style="magenta")
        table.add_column("Target", style="green")
        table.add_column("Depth", style="blue")

        for r in results:
            table.add_row(
                str(r.get("source", "")),
                str(r.get("relation", "")),
                str(r.get("target", "")),
                str(r.get("depth", "")),
            )
        console.print(table)
    finally:
        _close_engine(engine)


# ── debug / stats ─────────────────────────────────────────────────────────


@app.command("explain")
def explain(
    query: str = typer.Argument(..., help="Query to explain"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database file path"),
) -> None:
    """Explain how a query would be processed."""
    engine = _make_engine(Path(db_path) if db_path else None)
    try:
        trace = asyncio.run(engine.explain_query(query))
        console.print_json(json.dumps(trace, default=str, indent=2))
    finally:
        _close_engine(engine)


@app.command("stats")
def stats(
    db_path: Optional[str] = typer.Option(None, "--db", help="Database file path"),
) -> None:
    """Show context database statistics."""
    engine = _make_engine(Path(db_path) if db_path else None)
    try:
        metrics = asyncio.run(engine.get_metrics())
        console.print_json(json.dumps(metrics, default=str, indent=2))
    finally:
        _close_engine(engine)


@app.command("export")
def export(
    scope: Optional[str] = typer.Option(None, "--scope", help="Filter by scope"),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json|markdown"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database file path"),
) -> None:
    """Export context entries."""
    engine = _make_engine(Path(db_path) if db_path else None)
    try:
        context_scope = ContextScope(scope) if scope else None
        entries = asyncio.run(engine.list_entries(scope=context_scope, limit=10000))

        if format == "json":
            data = [
                {
                    "id": str(e.id),
                    "type": e.context_type.value,
                    "scope": e.scope.value,
                    "content": e.l2_full,
                    "tags": e.tags,
                    "status": e.status.value,
                }
                for e in entries
            ]
            result = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        else:
            lines = []
            for e in entries:
                lines.append(f"## {e.id}")
                lines.append(f"- Type: {e.context_type.value}")
                lines.append(f"- Scope: {e.scope.value}")
                lines.append(f"- Tags: {', '.join(e.tags or [])}")
                lines.append(f"\n{e.l2_full}\n")
            result = "\n".join(lines)

        if output:
            Path(output).write_text(result, encoding="utf-8")
            console.print(f"[green]✓[/green] Exported to: {output}")
        else:
            console.print(result)
    finally:
        _close_engine(engine)


if __name__ == "__main__":
    app()

