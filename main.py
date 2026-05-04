"""AgentBase — Context Database for AI Agents.

Unified entry point that launches the most common AgentBase interfaces.

Usage:
    # Launch web dashboard (default)
    python main.py web [db_path] [port]

    # Launch MCP server
    python main.py mcp

    # Launch CLI
    python main.py cli [command] [args...]

    # Quick start (initialize + demo)
    python main.py demo
"""

from __future__ import annotations

import sys


def main() -> None:
    """Route to the appropriate AgentBase interface."""
    if len(sys.argv) < 2:
        _print_help()
        sys.exit(0)

    command = sys.argv[1]
    rest = sys.argv[2:]

    if command == "web":
        _launch_web(rest)
    elif command == "mcp":
        _launch_mcp(rest)
    elif command == "cli":
        _launch_cli(rest)
    elif command == "demo":
        _launch_demo(rest)
    else:
        print(f"Unknown command: {command}")
        _print_help()
        sys.exit(1)


def _print_help() -> None:
    """Print usage help."""
    print(
        "AgentBase — Context Database for AI Agents\n"
        "\n"
        "Usage:\n"
        "  python main.py web [db_path] [port]  Start the web dashboard\n"
        "  python main.py mcp                   Start the MCP server\n"
        "  python main.py cli [command] [...]   Run a CLI command\n"
        "  python main.py demo                  Run a quick demo\n"
    )


def _launch_web(args: list[str]) -> None:
    """Start the AgentBase web dashboard."""
    from agentbase_web.app import main as web_main

    # Pass remaining args so web_main can pick up db_path and port
    if args:
        sys.argv = ["agentbase-web"] + args
    web_main()


def _launch_mcp(args: list[str]) -> None:
    """Start the AgentBase MCP server."""
    from agentbase_mcp.server import run_server

    db_path = args[0] if args else "agentbase.db"
    run_server(db_path=db_path)


def _launch_cli(args: list[str]) -> None:
    """Run an AgentBase CLI command."""
    from agentbase_cli.main import app

    sys.argv = ["agentbase"] + (args if args else ["--help"])
    app()


def _launch_demo(args: list[str]) -> None:
    """Run a quick demonstration of AgentBase capabilities."""
    import asyncio
    import tempfile
    from pathlib import Path

    from agentbase import AgentBase

    async def _demo() -> None:
        tmp = Path(tempfile.mkdtemp())
        db_path = tmp / "demo.db"
        db = AgentBase(path=db_path)
        await db.initialize()

        print("=" * 60)
        print("  AgentBase Demo — Context Database for AI Agents")
        print("=" * 60)

        # Add memories
        m1 = await db.add_memory(
            "User prefers Python 3.12 for backend development",
            category="preference",
            tags=["python", "backend"],
        )
        print(f"\n[Memory] Added: {m1.l2_full[:60]}... (id={m1.id})")

        m2 = await db.add_memory(
            "User's favorite editor is VS Code with vim keybindings",
            category="preference",
            tags=["editor", "vim"],
        )
        print(f"[Memory] Added: {m2.l2_full[:60]}... (id={m2.id})")

        # Add resource
        r1 = await db.add_resource(
            url="https://docs.python.org/3/",
            content="Python 3 official documentation",
        )
        print(f"[Resource] Added: {r1.l2_full[:60]}... (id={r1.id})")

        # Add skill
        s1 = await db.add_skill(
            tool_name="web_search",
            description="Search the web for up-to-date information",
        )
        print(f"[Skill] Added: {s1.l2_full[:60]}... (id={s1.id})")

        # Search
        print("\n--- Search: 'Python preferences' ---")
        results = await db.find("Python preferences", top_k=5)
        for r in results:
            print(f"  [{r.entry.context_type.value}] {r.entry.l2_full[:50]}... (score={r.score:.3f})")

        # Count
        count = await db.count()
        print(f"\nTotal entries: {count}")

        await db.close()
        print("\nDemo complete!")

    asyncio.run(_demo())


if __name__ == "__main__":
    main()
