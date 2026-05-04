"""Abstract embedding interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class AbstractEmbedder(ABC):
    """Abstract embedding interface for AgentBase."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensions."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name used for embedding."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding vector for a single text."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts."""
