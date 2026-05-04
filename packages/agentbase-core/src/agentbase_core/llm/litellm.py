"""LiteLLM-based LLM implementation."""

from __future__ import annotations

import json
import logging
from typing import Any

from ..exceptions import LLMError
from ..models.config import LLMConfig
from .base import AbstractLLM

logger = logging.getLogger(__name__)


class LiteLLMChat(AbstractLLM):
    """LLM implementation with OpenAI SDK preferred, litellm as fallback.

    When ``api_base`` is set (e.g. DashScope compatible-mode endpoint), the
    direct OpenAI SDK path is used first because litellm's provider routing
    can hang for non-standard providers.  When ``api_base`` is not set,
    litellm is tried first so that its automatic provider detection works
    for well-known OpenAI / Anthropic / etc. models.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        # Reuse a single AsyncOpenAI client across calls
        self._openai_client: Any | None = None

    def _get_openai_client(self) -> Any:
        if self._openai_client is None:
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(
                api_key=self._config.api_key,
                base_url=self._config.api_base,
            )
        return self._openai_client

    async def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        """Generate a text completion.

        Priority when ``api_base`` is set: OpenAI SDK → litellm.
        Priority when ``api_base`` is unset: litellm → OpenAI SDK.
        """
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})

        # Strip provider prefix for OpenAI SDK (e.g. "dashscope/glm-5.1" → "glm-5.1")
        model_name = self._config.model.split("/", 1)[-1]
        temperature = kwargs.get("temperature", self._config.temperature)
        max_tokens = kwargs.get("max_tokens", self._config.max_tokens)

        # --- When api_base is set, prefer direct OpenAI SDK ---
        if self._config.api_base:
            try:
                client = self._get_openai_client()
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.debug(f"OpenAI SDK call failed, trying litellm: {e}")

        # --- litellm (primary when no api_base, fallback when api_base set) ---
        try:
            from litellm import acompletion
            response = await acompletion(
                model=self._config.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_base=self._config.api_base,
                api_key=self._config.api_key,
                timeout=kwargs.get("timeout", 60),
            )
            return response.choices[0].message.content
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"litellm call failed: {e}")

        # --- Last resort: OpenAI SDK even without api_base ---
        if not self._config.api_base:
            try:
                client = self._get_openai_client()
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                raise LLMError(f"LLM completion failed: {e}") from e

        raise LLMError(f"LLM completion failed for model={self._config.model}")

    async def complete_json(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> Any:
        """Generate a JSON completion via litellm."""
        text = await self.complete(prompt, system=system, **kwargs)

        # Try to parse JSON from the response
        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (code fences)
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON array/object in the text
            for start_char, end_char in [("[", "]"), ("{", "}")]:
                start = text.find(start_char)
                end = text.rfind(end_char)
                if start != -1 and end != -1 and end > start:
                    try:
                        return json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        continue
            raise LLMError(f"Failed to parse LLM response as JSON: {text[:200]}")
