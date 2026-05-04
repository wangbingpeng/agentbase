"""Shared test fixtures for AgentBase tests."""

import asyncio
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from agentbase_core.engine import AgentBaseEngine
from agentbase_core.models.config import AgentBaseConfig, GraphConfig, ObservabilityConfig, SessionConfig
from agentbase_core.store.connection import ConnectionPool
from agentbase_core.store.sqlite_store import SQLiteStore


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def tmp_db_path(tmp_path):
    """Provide a temporary database path."""
    return tmp_path / "test.db"


@pytest_asyncio.fixture
async def db_config(tmp_db_path):
    """Provide a test database config."""
    return AgentBaseConfig(
        data_dir=tmp_db_path.parent,
        db_filename=tmp_db_path.name,
    )


@pytest_asyncio.fixture
async def engine(tmp_db_path):
    """Provide an initialized AgentBaseEngine with a temporary database (feature flags OFF)."""
    eng = AgentBaseEngine(db_path=tmp_db_path)
    await eng.initialize()
    yield eng
    await eng.close()


@pytest_asyncio.fixture
async def full_engine(tmp_path):
    """Provide an initialized AgentBaseEngine with ALL feature flags enabled."""
    config = AgentBaseConfig(
        data_dir=tmp_path,
        db_filename="full_test.db",
        graph=GraphConfig(enabled=True),
        session=SessionConfig(enabled=True),
        observability=ObservabilityConfig(enabled=True),
    )
    eng = AgentBaseEngine(config=config)
    await eng.initialize()
    yield eng
    await eng.close()


@pytest_asyncio.fixture
async def pool(tmp_db_path):
    """Provide an initialized ConnectionPool."""
    p = ConnectionPool(tmp_db_path)
    await p.initialize()
    yield p
    await p.close_all()


@pytest_asyncio.fixture
async def store(pool):
    """Provide an initialized SQLiteStore."""
    s = SQLiteStore(pool)
    await s.initialize()
    yield s
