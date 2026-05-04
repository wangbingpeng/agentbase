"""AgentBase MCP Server — Model Context Protocol integration.

Exposes AgentBase as a set of MCP tools that can be used by AI agents
through the Model Context Protocol.

Available tools:
- add_memory: Add a memory context entry
- add_resource: Add a resource context entry
- add_skill: Add a skill context entry
- find_context: Search for context entries
- get_context: Get a specific context entry by ID
- delete_context: Delete a context entry
- list_context: List context entries
- add_entity: Add an entity to the knowledge graph
- find_entities: Find entities by name
- add_relation: Add a relation between entities
- graph_traverse: Traverse the knowledge graph
- create_session: Create a conversation session
- add_message: Add a message to a session
- get_session: Get session with messages
- commit_session: Commit a session and extract memories
- get_stats: Get context database statistics
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from agentbase import AgentBase
from agentbase_core.models import ContextScope, ContextType, MemoryCategory

logger = logging.getLogger(__name__)

# Global AgentBase instance
_db: AgentBase | None = None


def _get_db() -> AgentBase:
    """Get or create the AgentBase instance."""
    global _db
    if _db is None:
        raise RuntimeError("AgentBase not initialized. Call initialize() first.")
    return _db


async def initialize(db_path: str | Path = "agentbase.db") -> None:
    """Initialize the AgentBase database."""
    global _db
    _db = AgentBase(path=db_path)
    await _db.initialize()
    logger.info(f"AgentBase MCP server initialized: {db_path}")


async def shutdown() -> None:
    """Shutdown the AgentBase database."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None


# ------------------------------------------------------------------
# Tool definitions
# ------------------------------------------------------------------

_TOOL_DEFINITIONS = [
    Tool(
        name="add_memory",
        description="Add a memory context entry to the AgentBase context database",
        inputSchema={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Memory content"},
                "category": {"type": "string", "description": "Memory category (profile/preference/entity/event/case/pattern)", "default": "entity"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags", "default": []},
                "confidence": {"type": "number", "description": "Confidence score 0-1", "default": 1.0},
                "scope": {"type": "string", "description": "Scope (global/agent/project/session)", "default": "global"},
                "owner_id": {"type": "string", "description": "Agent/owner ID for scoped entries"},
            },
            "required": ["content"],
        },
    ),
    Tool(
        name="add_resource",
        description="Add a resource context entry (document, URL, tool spec, etc.)",
        inputSchema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Resource URL"},
                "content": {"type": "string", "description": "Resource content/description"},
                "format": {"type": "string", "description": "Resource format"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags"},
                "confidence": {"type": "number", "description": "Confidence 0-1", "default": 1.0},
                "scope": {"type": "string", "description": "Scope", "default": "global"},
            },
            "required": ["content"],
        },
    ),
    Tool(
        name="add_skill",
        description="Add a skill context entry (tool capability description)",
        inputSchema={
            "type": "object",
            "properties": {
                "tool_name": {"type": "string", "description": "Skill/tool name"},
                "description": {"type": "string", "description": "Skill description"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags"},
                "confidence": {"type": "number", "description": "Confidence 0-1", "default": 1.0},
            },
            "required": ["tool_name"],
        },
    ),
    Tool(
        name="find_context",
        description="Search for context entries by query text",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "description": "Max results", "default": 10},
                "type": {"type": "string", "description": "Filter by type (memory/resource/skill)"},
                "scope": {"type": "string", "description": "Filter by scope"},
                "token_budget": {"type": "integer", "description": "Token budget limit"},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="get_context",
        description="Get a specific context entry by ID",
        inputSchema={
            "type": "object",
            "properties": {
                "entry_id": {"type": "string", "description": "Entry ID"},
            },
            "required": ["entry_id"],
        },
    ),
    Tool(
        name="delete_context",
        description="Delete a context entry",
        inputSchema={
            "type": "object",
            "properties": {
                "entry_id": {"type": "string", "description": "Entry ID"},
            },
            "required": ["entry_id"],
        },
    ),
    Tool(
        name="add_entity",
        description="Add an entity to the knowledge graph",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Entity name"},
                "entity_type": {"type": "string", "description": "Entity type (person/project/concept/tool/event/organization)", "default": "concept"},
                "description": {"type": "string", "description": "Entity description"},
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="find_entities",
        description="Find entities by name",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Entity name to search"},
                "entity_type": {"type": "string", "description": "Filter by entity type"},
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="add_relation",
        description="Add a relation between two entities in the knowledge graph",
        inputSchema={
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "description": "Source entity ID"},
                "target_id": {"type": "string", "description": "Target entity ID"},
                "predicate": {"type": "string", "description": "Relation predicate (e.g., uses, works_on, belongs_to)"},
            },
            "required": ["source_id", "target_id", "predicate"],
        },
    ),
    Tool(
        name="graph_traverse",
        description="Traverse the knowledge graph starting from an entity",
        inputSchema={
            "type": "object",
            "properties": {
                "entity_name": {"type": "string", "description": "Starting entity name"},
                "depth": {"type": "integer", "description": "Traversal depth", "default": 2},
            },
            "required": ["entity_name"],
        },
    ),
    Tool(
        name="create_session",
        description="Create a new conversation session",
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent ID", "default": "default"},
                "project": {"type": "string", "description": "Project name"},
            },
            "required": [],
        },
    ),
    Tool(
        name="add_message",
        description="Add a message to a conversation session",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID"},
                "role": {"type": "string", "description": "Message role (user/assistant/tool/system)"},
                "content": {"type": "string", "description": "Message content"},
            },
            "required": ["session_id", "role", "content"],
        },
    ),
    Tool(
        name="commit_session",
        description="Commit a session: compress messages and extract memories",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID"},
                "mode": {"type": "string", "description": "Commit mode (full/archive_only/extract_only)", "default": "full"},
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="get_stats",
        description="Get AgentBase context database statistics",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
]


# ------------------------------------------------------------------
# Tool handler
# ------------------------------------------------------------------


async def _handle_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle a tool call."""
    db = _get_db()

    try:
        if name == "add_memory":
            category = None
            if arguments.get("category"):
                try:
                    category = MemoryCategory(arguments["category"])
                except ValueError:
                    pass
            scope = ContextScope(arguments.get("scope", "global"))
            entry = await db.add_memory(
                content=arguments["content"],
                category=category,
                tags=arguments.get("tags", []),
                confidence=arguments.get("confidence", 1.0),
                scope=scope,
                owner_id=arguments.get("owner_id"),
            )
            return [TextContent(type="text", text=json.dumps({
                "id": entry.id, "type": entry.context_type.value,
                "scope": entry.scope.value, "status": entry.status.value,
            }))]

        elif name == "add_resource":
            scope = ContextScope(arguments.get("scope", "global"))
            entry = await db.add_resource(
                url=arguments.get("url"),
                content=arguments.get("content", ""),
                format=arguments.get("format"),
                tags=arguments.get("tags", []),
                confidence=arguments.get("confidence", 1.0),
                scope=scope,
            )
            return [TextContent(type="text", text=json.dumps({
                "id": entry.id, "type": entry.context_type.value, "scope": entry.scope.value,
            }))]

        elif name == "add_skill":
            entry = await db.add_skill(
                tool_name=arguments["tool_name"],
                description=arguments.get("description", ""),
                tags=arguments.get("tags", []),
                confidence=arguments.get("confidence", 1.0),
            )
            return [TextContent(type="text", text=json.dumps({
                "id": entry.id, "tool_name": entry.skill_tool_name, "type": entry.context_type.value,
            }))]

        elif name == "find_context":
            ct = ContextType(arguments["type"]) if arguments.get("type") else None
            sc = ContextScope(arguments["scope"]) if arguments.get("scope") else None
            results = await db.find(
                query=arguments["query"],
                top_k=arguments.get("top_k", 10),
                context_type=ct,
                scope=sc,
                token_budget=arguments.get("token_budget"),
            )
            items = []
            for r in results:
                items.append({
                    "id": r.entry.id,
                    "type": r.entry.context_type.value,
                    "scope": r.entry.scope.value,
                    "score": round(r.score, 4),
                    "l0": r.entry.l0_abstract,
                    "l1": r.entry.l1_overview,
                    "l2": (r.entry.l2_full or "")[:500],
                })
            return [TextContent(type="text", text=json.dumps(items, ensure_ascii=False))]

        elif name == "get_context":
            entry = await db.get(arguments["entry_id"])
            if entry is None:
                return [TextContent(type="text", text=json.dumps({"error": "Not found"}))]
            return [TextContent(type="text", text=json.dumps({
                "id": entry.id, "type": entry.context_type.value,
                "scope": entry.scope.value, "status": entry.status.value,
                "confidence": entry.confidence, "tags": entry.tags,
                "l0": entry.l0_abstract, "l1": entry.l1_overview,
                "l2": entry.l2_full,
            }, ensure_ascii=False))]

        elif name == "delete_context":
            ok = await db.delete(arguments["entry_id"])
            return [TextContent(type="text", text=json.dumps({"deleted": ok}))]

        elif name == "add_entity":
            entity = await db.add_entity(
                name=arguments["name"],
                entity_type=arguments.get("entity_type", "concept"),
                description=arguments.get("description", ""),
            )
            return [TextContent(type="text", text=json.dumps({
                "id": entity.id, "name": entity.name, "type": entity.entity_type,
            }))]

        elif name == "find_entities":
            entities = await db.find_entities(
                name=arguments["name"],
                entity_type=arguments.get("entity_type"),
            )
            items = [{"id": e.id, "name": e.name, "type": e.entity_type, "desc": e.description} for e in entities]
            return [TextContent(type="text", text=json.dumps(items, ensure_ascii=False))]

        elif name == "add_relation":
            rel = await db.add_relation(
                source_id=arguments["source_id"],
                target_id=arguments["target_id"],
                predicate=arguments["predicate"],
            )
            return [TextContent(type="text", text=json.dumps({"id": rel.id, "predicate": rel.predicate}))]

        elif name == "graph_traverse":
            results = await db.graph_traversal(
                entity_name=arguments["entity_name"],
                depth=arguments.get("depth", 2),
            )
            return [TextContent(type="text", text=json.dumps(results, ensure_ascii=False))]

        elif name == "create_session":
            session = await db.create_session(
                agent_id=arguments.get("agent_id", "default"),
                project=arguments.get("project"),
            )
            return [TextContent(type="text", text=json.dumps({"id": session.id, "agent_id": session.agent_id}))]

        elif name == "add_message":
            msg = await db.add_message(
                session_id=arguments["session_id"],
                role=arguments["role"],
                content=arguments["content"],
            )
            return [TextContent(type="text", text=json.dumps({"id": msg.id, "role": msg.role}))]

        elif name == "commit_session":
            memories = await db.commit_session(
                session_id=arguments["session_id"],
                mode=arguments.get("mode", "full"),
            )
            return [TextContent(type="text", text=json.dumps({"extracted_count": len(memories)}))]

        elif name == "get_stats":
            metrics = await db.get_metrics()
            return [TextContent(type="text", text=json.dumps(metrics))]

        else:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    except Exception as e:
        logger.error(f"Tool error: {name} - {e}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


# ------------------------------------------------------------------
# Server creation
# ------------------------------------------------------------------


def create_server(db_path: str | Path = "agentbase.db") -> Server:
    """Create and configure the MCP server."""
    server = Server("agentbase")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return _TOOL_DEFINITIONS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        return await _handle_tool(name, arguments)

    return server


async def run_server(db_path: str | Path = "agentbase.db") -> None:
    """Run the MCP server with stdio transport."""
    await initialize(db_path)
    server = create_server(db_path)

    async with stdio_server() as (read_stream, write_stream):
        try:
            await server.run(read_stream, write_stream, server.create_initialization_options())
        finally:
            await shutdown()


def main() -> None:
    """Entry point for the MCP server CLI."""
    import sys
    from pathlib import Path as _Path

    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        # Default to ~/.agentbase/agentbase.db for consistent behavior
        # regardless of the caller's working directory
        db_path = str(_Path.home() / ".agentbase" / "agentbase.db")
    asyncio.run(run_server(db_path))


if __name__ == "__main__":
    main()
