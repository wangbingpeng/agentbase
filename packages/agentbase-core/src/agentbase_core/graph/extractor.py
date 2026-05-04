"""EntityExtractor — extract entities and relations from text using LLM."""

from __future__ import annotations

import logging
from typing import Any

from ..exceptions import LLMError
from ..llm.base import AbstractLLM
from ..models.entity import Entity, Relation

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = """你是一个实体关系提取器。从给定文本中提取实体和关系。

Rules:
- 实体类型：person, project, concept, tool, event, organization
- 关系谓词：prefers, works_on, depends_on, contains, belongs_to, uses, located_at, happens_at
- 每个实体必须有 name 和 type
- 每个关系必须有 source(实体名), target(实体名), predicate
- 置信度 0-1
- 返回 JSON

Text:
{text}

Return JSON:
{{
  "entities": [
    {{"name": "...", "type": "...", "description": "..."}}
  ],
  "relations": [
    {{"source": "...", "target": "...", "predicate": "...", "confidence": 0.9}}
  ]
}}"""


class EntityExtractor:
    """Extract entities and relations from text using LLM."""

    def __init__(self, llm: AbstractLLM) -> None:
        self._llm = llm

    async def extract(self, text: str) -> tuple[list[Entity], list[dict]]:
        """Extract entities and relations from text.

        Returns (entities, raw_relations) where raw_relations are dicts
        with source/target names that need resolution to IDs.
        """
        try:
            result = await self._llm.complete_json(
                prompt=_EXTRACT_PROMPT.format(text=text),
                system="You are an entity-relation extractor.",
            )
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return [], []

        if not isinstance(result, dict):
            return [], []

        entities = []
        for item in result.get("entities", []):
            if not isinstance(item, dict) or not item.get("name"):
                continue
            entities.append(Entity(
                name=item["name"],
                entity_type=item.get("type", "concept"),
                description=item.get("description", ""),
            ))

        raw_relations = []
        for item in result.get("relations", []):
            if not isinstance(item, dict) or not item.get("source") or not item.get("target"):
                continue
            raw_relations.append({
                "source": item["source"],
                "target": item["target"],
                "predicate": item.get("predicate", "related_to"),
                "confidence": item.get("confidence", 0.8),
            })

        return entities, raw_relations
