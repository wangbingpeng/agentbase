"""MarkdownExporter — export SQLite context to human-readable Markdown."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..models.context_entry import ContextEntry

logger = logging.getLogger(__name__)


class MarkdownExporter:
    """Export SQLite context entries to human-readable Markdown files."""

    async def export_entry(self, entry: ContextEntry, path: Path) -> None:
        """Export a single entry to a Markdown file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "context_type": entry.context_type.value,
            "scope": entry.scope.value,
            "tags": entry.tags,
            "confidence": entry.confidence,
            "l0": entry.l0_abstract,
            "l1": entry.l1_overview,
        }
        content = f"""## [{entry.id}] {entry.l0_abstract or 'Untitled'}

{entry.l2_full}

<!-- meta: {json.dumps(meta, ensure_ascii=False)} -->
"""
        path.write_text(content, encoding="utf-8")

    async def export_entries(self, entries: list[ContextEntry], output_dir: Path) -> int:
        """Export multiple entries to Markdown files in a directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for entry in entries:
            filename = f"{entry.id}.md"
            await self.export_entry(entry, output_dir / filename)
            count += 1
        return count
