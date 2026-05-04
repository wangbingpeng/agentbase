"""Phase 8 Hardening tests — feature flags, reindex, cleanup, YAML config."""

import pytest
import pytest_asyncio

from agentbase_core.engine import AgentBaseEngine
from agentbase_core.exceptions import ConfigError
from agentbase_core.models.config import (
    AgentBaseConfig,
    GraphConfig,
    ObservabilityConfig,
    SessionConfig,
)


@pytest.mark.asyncio
class TestFeatureFlags:
    """Test feature flag enforcement."""

    async def test_graph_enabled_by_default(self, engine):
        """Graph should be enabled by default (graceful degradation if no LLM)."""
        assert engine.config.graph.enabled is True
        # EntityService should be available even without LLM (CRUD-only mode)
        assert engine.entity_service is not None

    async def test_session_enabled_by_default(self, engine):
        """Session should be enabled by default (graceful degradation if no LLM)."""
        assert engine.config.session.enabled is True
        assert engine.session_service is not None

    async def test_graph_explicitly_disabled(self, tmp_path):
        """Graph operations should fail when explicitly disabled."""
        from agentbase_core.models.config import IndexConfig
        config = AgentBaseConfig(
            data_dir=tmp_path,
            db_filename="test.db",
            graph=GraphConfig(enabled=False),
        )
        eng = AgentBaseEngine(config=config)
        await eng.initialize()
        try:
            with pytest.raises(ConfigError, match="Graph feature is disabled"):
                _ = eng.entity_service
        finally:
            await eng.close()

    async def test_session_explicitly_disabled(self, tmp_path):
        """Session operations should fail when explicitly disabled."""
        config = AgentBaseConfig(
            data_dir=tmp_path,
            db_filename="test.db",
            session=SessionConfig(enabled=False),
        )
        eng = AgentBaseEngine(config=config)
        await eng.initialize()
        try:
            with pytest.raises(ConfigError, match="Session feature is disabled"):
                _ = eng.session_service
        finally:
            await eng.close()

    async def test_observability_disabled_by_default(self, engine):
        """Observability should fail when disabled."""
        with pytest.raises(ConfigError, match="Observability feature is disabled"):
            _ = engine.metrics

    async def test_graph_enabled(self, full_engine):
        """Graph operations should work when enabled."""
        assert full_engine.entity_service is not None

    async def test_session_enabled(self, full_engine):
        """Session operations should work when enabled."""
        assert full_engine.session_service is not None

    async def test_observability_enabled(self, full_engine):
        """Observability should work when enabled."""
        assert full_engine.metrics is not None


@pytest.mark.asyncio
class TestReindex:
    """Test index rebuild."""

    async def test_reindex_fts(self, engine):
        """FTS reindex should work and return count."""
        await engine.add_memory("Test content for reindex")
        result = await engine.reindex()
        assert "fts" in result
        assert result["fts"] >= 1

    async def test_reindex_preserves_search(self, engine):
        """Reindex should preserve searchability."""
        await engine.add_memory("Unique keyword: xyzwq123")
        # Search before reindex
        results_before = await engine.find("xyzwq123")
        assert len(results_before) >= 1

        # Reindex
        await engine.reindex()

        # Search after reindex
        results_after = await engine.find("xyzwq123")
        assert len(results_after) >= 1


@pytest.mark.asyncio
class TestCleanup:
    """Test data cleanup."""

    async def test_cleanup_traces(self, full_engine):
        """Cleanup should delete old traces."""
        # Even with no traces, cleanup should succeed
        result = await full_engine.cleanup(traces_older_than_days=30)
        assert "traces" in result
        assert result["traces"] == 0

    async def test_cleanup_deleted_entries(self, engine):
        """Cleanup should purge soft-deleted entries older than threshold."""
        entry = await engine.add_memory("To be deleted")
        await engine.delete(entry.id)

        # Cleanup with 0 days threshold (purge all deleted)
        result = await engine.cleanup(deleted_older_than_days=0)
        assert "deleted_entries" in result

    async def test_cleanup_no_criteria(self, engine):
        """Cleanup with no criteria should return empty dict."""
        result = await engine.cleanup()
        assert result == {}


class TestYAMLConfig:
    """Test YAML configuration loading and saving."""

    def test_to_yaml_and_from_yaml(self, tmp_path):
        """Config round-trips through YAML."""
        config = AgentBaseConfig(
            data_dir=tmp_path,
            graph=GraphConfig(enabled=True),
            session=SessionConfig(enabled=True),
            observability=ObservabilityConfig(enabled=True),
        )
        yaml_path = tmp_path / "agentbase.yaml"
        config.to_yaml(yaml_path)

        loaded = AgentBaseConfig.from_yaml(yaml_path)
        assert loaded.graph.enabled is True
        assert loaded.session.enabled is True
        assert loaded.observability.enabled is True

    def test_from_yaml_missing_file(self, tmp_path):
        """Loading from missing file should use defaults."""
        config = AgentBaseConfig.from_yaml(tmp_path / "nonexistent.yaml")
        assert config.graph.enabled is True
        assert config.session.enabled is True

    def test_env_vars_work(self, monkeypatch):
        """Environment variables should configure AgentBaseConfig."""
        monkeypatch.setenv("AGENTBASE_GRAPH__ENABLED", "true")
        config = AgentBaseConfig()
        # pydantic-settings auto-reads env vars on construction
        assert config.graph.enabled is True
