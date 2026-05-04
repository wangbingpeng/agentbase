"""Layer generation — L0/L1 async generation with fallback."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..exceptions import LLMError
from ..models.context_entry import ContextEntry

logger = logging.getLogger(__name__)

# --- L0/L1 generation prompts ---

_L0_SYSTEM_PROMPT = """你是一个内容摘要生成器。为给定的内容生成一句话摘要。

Rules:
- 摘要不超过 50 字
- 只包含最核心的信息
- 使用陈述句

Return only the abstract, nothing else."""

_L1_SYSTEM_PROMPT = """你是一个内容概要生成器。为给定的内容生成结构化概要。

Rules:
- 概要不超过 300 字
- 包含关键实体、关系和结论
- 保留时间、数量等关键事实
- 使用简洁的要点格式

Return only the overview, nothing else."""


class LayerGenerator:
    """Generate L0/L1 layers for context entries."""

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm

    async def generate_l0(self, content: str) -> str | None:
        """Generate L0 abstract from content."""
        if self._llm is None:
            return None

        try:
            result = await self._llm.complete(
                prompt=content,
                system=_L0_SYSTEM_PROMPT,
            )
            # Truncate to 50 chars if needed
            if len(result) > 50:
                result = result[:50]
            return result.strip()
        except LLMError as e:
            logger.warning(f"L0 generation failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"L0 generation unexpected error: {e}")
            return None

    async def generate_l1(self, content: str) -> str | None:
        """Generate L1 overview from content."""
        if self._llm is None:
            return None

        try:
            result = await self._llm.complete(
                prompt=content,
                system=_L1_SYSTEM_PROMPT,
            )
            # Truncate to 300 chars if needed
            if len(result) > 300:
                result = result[:300]
            return result.strip()
        except LLMError as e:
            logger.warning(f"L1 generation failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"L1 generation unexpected error: {e}")
            return None

    async def generate_layers(self, entry: ContextEntry) -> ContextEntry:
        """Generate L0/L1 for an entry, with truncation fallback."""
        content = entry.l2_full
        if not content:
            return entry

        # Try LLM generation
        l0 = await self.generate_l0(content)
        l1 = await self.generate_l1(content)

        if l0:
            entry.l0_abstract = l0
        else:
            # Fallback: truncation
            entry.l0_abstract = content[:100] + "..." if len(content) > 100 else content

        if l1:
            entry.l1_overview = l1
        else:
            # Fallback: truncation
            entry.l1_overview = content[:500] + "..." if len(content) > 500 else content

        return entry
