"""Abstract base classes for index backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models.context_entry import ContextEntry
from ..models.query import SearchResult


class AbstractIndex(ABC):
    """Abstract index interface."""

    @abstractmethod
    async def add(self, entry: ContextEntry) -> None:
        """Index a context entry."""

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 10,
        context_type: str | None = None,
        scope: str | None = None,
        owner_id: str | None = None,
        fts_column: str | None = None,
    ) -> list[SearchResult]:
        """Search the index."""

    @abstractmethod
    async def remove(self, entry_id: str) -> None:
        """Remove an entry from the index."""

    @abstractmethod
    async def count(self) -> int:
        """Return the number of indexed entries."""
