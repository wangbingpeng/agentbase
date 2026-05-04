"""ConflictResolver — resolve fact conflicts between existing and new facts."""

from __future__ import annotations

import logging
from enum import Enum

from ..llm.base import AbstractLLM

logger = logging.getLogger(__name__)


class ConflictResolution(str, Enum):
    MERGE = "merge"
    REPLACE = "replace"
    SKIP = "skip"


class ConflictResolver:
    """Resolve fact conflicts using LLM judgment."""

    _PROMPT = """判断新事实与已有事实的关系。

已有事实: {existing_fact}
新事实: {new_fact}

判断：
- "merge": 两者兼容，合并描述
- "replace": 新事实更准确，替代旧事实
- "skip": 旧事实更可靠，跳过新事实

Return only one word: merge/replace/skip"""

    def __init__(self, llm: AbstractLLM) -> None:
        self._llm = llm

    async def resolve(self, existing: str, new: str) -> ConflictResolution:
        """Resolve a conflict between existing and new fact."""
        try:
            result = await self._llm.complete(
                prompt=self._PROMPT.format(existing_fact=existing, new_fact=new),
            )
            result = result.strip().lower()
            if result in ("merge", "replace", "skip"):
                return ConflictResolution(result)
        except Exception as e:
            logger.warning(f"Conflict resolution failed: {e}")

        # Default: skip (preserve existing)
        return ConflictResolution.SKIP

    @staticmethod
    def resolve_by_rule(existing: str, new: str) -> ConflictResolution:
        """Simple rule-based conflict resolution (no LLM)."""
        # If new fact is longer and contains existing, likely a replacement
        if existing in new and len(new) > len(existing):
            return ConflictResolution.REPLACE
        # If they're very similar, merge
        if existing.lower() == new.lower():
            return ConflictResolution.SKIP
        # Default: merge (additive)
        return ConflictResolution.MERGE
