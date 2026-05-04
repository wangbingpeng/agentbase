"""Unit tests for agentbase-mcp server tool handler."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import pytest_asyncio

from agentbase_mcp import server as mcp_server


@pytest_asyncio.fixture
async def mcp_db(tmp_path: Path):
    """Initialize a fresh MCP-backed AgentBase for each test."""
    db_path = tmp_path / "mcp_test.db"
    await mcp_server.initialize(db_path)
    try:
        yield db_path
    finally:
        await mcp_server.shutdown()


class TestServerLifecycle:
    async def test_initialize_creates_db(self, tmp_path: Path):
        db_path = tmp_path / "lifecycle.db"
        await mcp_server.initialize(db_path)
        try:
            assert mcp_server._db is not None
        finally:
            await mcp_server.shutdown()
        assert mcp_server._db is None

    async def test_get_db_before_init_raises(self):
        # Ensure clean slate
        if mcp_server._db is not None:
            await mcp_server.shutdown()
        with pytest.raises(RuntimeError):
            mcp_server._get_db()

    def test_create_server_returns_server(self, tmp_path: Path):
        srv = mcp_server.create_server(tmp_path / "srv.db")
        assert srv is not None
        assert srv.name == "agentbase"


class TestToolDefinitions:
    def test_tool_definitions_nonempty(self):
        assert len(mcp_server._TOOL_DEFINITIONS) > 0

    def test_required_tools_registered(self):
        names = {t.name for t in mcp_server._TOOL_DEFINITIONS}
        required = {
            "add_memory",
            "add_resource",
            "add_skill",
            "find_context",
            "get_context",
            "delete_context",
            "get_stats",
        }
        assert required.issubset(names), f"missing tools: {required - names}"


class TestToolHandler:
    """Exercise the async tool-call handler directly."""

    async def test_add_and_find_memory(self, mcp_db: Path):
        # add_memory
        result = await mcp_server._handle_tool(
            "add_memory",
            {"content": "用户喜欢Python和咖啡", "tags": ["python", "coffee"]},
        )
        assert len(result) == 1
        payload = json.loads(result[0].text)
        assert "id" in payload
        assert payload.get("type") == "memory"

        # find_context
        result = await mcp_server._handle_tool(
            "find_context",
            {"query": "Python", "top_k": 5},
        )
        assert len(result) == 1
        # Must return valid JSON (either list of results or {"results": [...]})
        body = json.loads(result[0].text)
        assert body is not None

    async def test_get_stats(self, mcp_db: Path):
        result = await mcp_server._handle_tool("get_stats", {})
        assert len(result) == 1
        body = json.loads(result[0].text)
        assert isinstance(body, dict)

    async def test_unknown_tool_returns_error(self, mcp_db: Path):
        result = await mcp_server._handle_tool("no_such_tool_xyz", {})
        assert len(result) == 1
        body = json.loads(result[0].text)
        assert "error" in body

    async def test_invalid_arguments_handled(self, mcp_db: Path):
        # add_memory without required "content" should return error payload, not raise
        result = await mcp_server._handle_tool("add_memory", {})
        assert len(result) == 1
        body = json.loads(result[0].text)
        assert "error" in body
