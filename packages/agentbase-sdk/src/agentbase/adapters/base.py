"""Abstract base class for all AgentBase adapters."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentbase import AgentBase


class BaseAdapter(ABC):
    """Base class that holds a reference to an AgentBase instance.

    Subclasses implement framework-specific interfaces by delegating
    all storage/retrieval operations to the underlying AgentBase instance.
    """

    def __init__(self, db: AgentBase) -> None:
        self._db = db

    @property
    def db(self) -> AgentBase:
        """Return the underlying AgentBase instance."""
        return self._db
