"""TokenBudget — token budget manager for retrieval result trimming."""

from __future__ import annotations


class TokenBudget:
    """Token budget manager — controls how much context content can be loaded.

    Uses character-based estimation: 1 Chinese char ≈ 1 token,
    1 English word ≈ 1.3 tokens. For simplicity, we estimate
    tokens ≈ len(text) // 2 for mixed content.
    """

    def __init__(self, budget: int = 4000) -> None:
        self._budget = budget
        self._used = 0

    def can_load(self, estimated_tokens: int) -> bool:
        """Check if we can load more content within the budget."""
        return self._used + estimated_tokens <= self._budget

    def allocate(self, tokens: int) -> None:
        """Allocate token budget."""
        self._used += tokens

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text string."""
        if not text:
            return 0
        # Simple heuristic: for mixed CJK/Latin, ~0.5 tokens per char
        return max(1, len(text) // 2)

    def try_allocate(self, text: str) -> bool:
        """Try to allocate budget for a text; returns True if successful."""
        est = self.estimate_tokens(text)
        if self.can_load(est):
            self.allocate(est)
            return True
        return False

    @property
    def remaining(self) -> int:
        return self._budget - self._used

    @property
    def utilization(self) -> float:
        return self._used / self._budget if self._budget > 0 else 0.0

    @property
    def budget(self) -> int:
        return self._budget

    @property
    def used(self) -> int:
        return self._used
