"""LiteLLM-based embedding implementation."""

from __future__ import annotations

import logging
from typing import Any

from ..exceptions import EmbeddingError
from ..models.config import EmbeddingConfig
from .base import AbstractEmbedder

# Ensure litellm drops unsupported params before any import
try:
    import litellm
    litellm.drop_params = True
except ImportError:
    pass

logger = logging.getLogger(__name__)


class LiteLLMEmbedder(AbstractEmbedder):
    """Embedding implementation using litellm for multi-provider support."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._dimensions = config.dimensions
        self._model_name = config.model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Tries litellm first; falls back to direct OpenAI SDK call
        when litellm raises UnsupportedParamsError (common with
        non-OpenAI providers like DashScope).
        """
        # --- Attempt 1: litellm ---
        try:
            import litellm
            from litellm import aembedding
            litellm.drop_params = True

            response = await aembedding(
                model=self._config.model,
                input=texts,
                api_base=self._config.api_base,
                api_key=self._config.api_key,
                dimensions=self._config.dimensions,
            )
            return [item["embedding"] for item in response.data]
        except ImportError:
            pass  # Fall through to direct SDK
        except Exception:
            pass  # Fall through to direct SDK

        # --- Attempt 2: Direct OpenAI SDK (works with DashScope etc.) ---
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise EmbeddingError("Either litellm or openai package is required for embedding.")

        try:
            client = AsyncOpenAI(
                api_key=self._config.api_key,
                base_url=self._config.api_base,
            )
            # Strip provider prefix (e.g., "openai/text-embedding-v3" → "text-embedding-v3")
            model_name = self._config.model.split("/", 1)[-1]
            response = await client.embeddings.create(
                model=model_name,
                input=texts,
                dimensions=self._config.dimensions,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise EmbeddingError(f"Embedding generation failed: {e}") from e
