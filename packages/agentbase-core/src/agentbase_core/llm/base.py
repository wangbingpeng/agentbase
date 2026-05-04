"""Abstract LLM interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractLLM(ABC):
    """Abstract LLM interface for AgentBase."""

    @abstractmethod
    async def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        """Generate a text completion."""

    @abstractmethod
    async def complete_json(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> Any:
        """Generate a JSON completion."""
