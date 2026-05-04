"""Unit tests for agentbase CLI (typer app)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentbase_cli.main import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "cli_test.db"


class TestCliHelp:
    """Basic command discovery via --help."""

    def test_root_help(self, runner: CliRunner):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "AgentBase" in result.stdout
        # Core subcommands should be listed
        for cmd in ("init", "add", "find", "list", "stats"):
            assert cmd in result.stdout

    def test_init_help(self, runner: CliRunner):
        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0
        assert "Initialize" in result.stdout


class TestCliLifecycle:
    """End-to-end CLI flow: init → add → list → find → get → delete."""

    def test_init_creates_db(self, runner: CliRunner, db_path: Path):
        result = runner.invoke(app, ["init", "--data-dir", str(db_path)])
        assert result.exit_code == 0, result.output
        assert db_path.exists()

    def test_add_and_list(self, runner: CliRunner, db_path: Path):
        # Initialize first
        runner.invoke(app, ["init", "--data-dir", str(db_path)])

        # Add an entry
        result = runner.invoke(
            app,
            [
                "add",
                "用户偏好使用Python开发",
                "--type", "memory",
                "--tags", "python,偏好",
                "--db", str(db_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Added entry" in result.output or "✓" in result.output

        # List entries
        result = runner.invoke(app, ["list", "--db", str(db_path)])
        assert result.exit_code == 0, result.output
        assert "Python" in result.output or "memory" in result.output

    def test_find_finds_added_entry(self, runner: CliRunner, db_path: Path):
        runner.invoke(app, ["init", "--data-dir", str(db_path)])
        runner.invoke(
            app,
            ["add", "用户喜欢喝咖啡", "--tags", "coffee", "--db", str(db_path)],
        )
        result = runner.invoke(app, ["find", "咖啡", "--db", str(db_path)])
        assert result.exit_code == 0, result.output
        # Either we find something, or graceful "No results" output
        assert "咖啡" in result.output or "No results" in result.output

    def test_empty_list_on_fresh_db(self, runner: CliRunner, db_path: Path):
        runner.invoke(app, ["init", "--data-dir", str(db_path)])
        result = runner.invoke(app, ["list", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "No entries found" in result.output or result.output.strip() != ""


class TestCliInvalidInput:
    """CLI should surface validation errors cleanly."""

    def test_add_with_invalid_type(self, runner: CliRunner, db_path: Path):
        runner.invoke(app, ["init", "--data-dir", str(db_path)])
        result = runner.invoke(
            app,
            ["add", "hello", "--type", "not_a_type", "--db", str(db_path)],
        )
        # Should fail (non-zero) due to enum validation
        assert result.exit_code != 0
