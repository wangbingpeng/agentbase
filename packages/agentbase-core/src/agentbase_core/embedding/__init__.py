"""AgentBase embedding layer."""

from .base import AbstractEmbedder
from .litellm import LiteLLMEmbedder

__all__ = ["AbstractEmbedder", "LiteLLMEmbedder"]