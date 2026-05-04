"""Language-aware tokenizer module for FTS5 full-text search.

Provides jieba-based Chinese word segmentation for CJK queries,
auto language detection, and fallback to basic word splitting for
non-CJK languages (English, etc.).

Usage:
    from agentbase_core.index.tokenizer import tokenize_text, tokenize_query

    # For indexing (write path)
    fts_text = tokenize_text("用户偏好使用Python")

    # For querying (read path)
    query_tokens = tokenize_query("偏好Python")

Tokenizer options:
    - "auto" (default): detect CJK -> jieba, otherwise -> basic word split
    - "jieba": always use jieba
    - "char": CJK character-level fallback
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CJK detection
# ---------------------------------------------------------------------------

_CJK_RANGES = (
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Extension A
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0xAC00, 0xD7AF),   # Hangul Syllables
)


def _is_cjk(ch: str) -> bool:
    """Check if a character is CJK."""
    code = ord(ch)
    for lo, hi in _CJK_RANGES:
        if lo <= code <= hi:
            return True
    return False


def _has_cjk(text: str) -> bool:
    """Check if text contains any CJK characters."""
    return any(_is_cjk(ch) for ch in text)


# ---------------------------------------------------------------------------
# jieba integration (lazy import)
# ---------------------------------------------------------------------------

_jieba_available: bool | None = None
_jieba_initialized: bool = False


def _check_jieba() -> bool:
    """Check if jieba is available (cached)."""
    global _jieba_available
    if _jieba_available is None:
        try:
            import jieba  # noqa: F401
            _jieba_available = True
        except ImportError:
            _jieba_available = False
            logger.info(
                "jieba not installed, CJK search will use character-level fallback. "
                "Install with: pip install jieba"
            )
    return _jieba_available


def _init_jieba() -> None:
    """Initialize jieba (lazy, once)."""
    global _jieba_initialized
    if not _jieba_initialized:
        import jieba
        jieba.setLogLevel(logging.WARNING)
        _jieba_initialized = True


# ---------------------------------------------------------------------------
# Fallback tokenizer: CJK character-level + non-CJK word splitting
# ---------------------------------------------------------------------------

def _fallback_tokenize(text: str) -> list[str]:
    """Tokenize without jieba: split CJK into individual characters,
    keep non-CJK words intact.

    This is a conservative fallback — single CJK characters have low
    precision but guarantee recall.
    """
    tokens: list[str] = []
    cjk_run: list[str] = []
    non_cjk_run: list[str] = []

    for ch in text:
        if _is_cjk(ch):
            if non_cjk_run:
                word = "".join(non_cjk_run).strip("-")
                if word:
                    tokens.append(word)
                non_cjk_run = []
            cjk_run.append(ch)
        else:
            if cjk_run:
                # Emit individual CJK chars + bigrams for recall
                for c in cjk_run:
                    tokens.append(c)
                for i in range(len(cjk_run) - 1):
                    tokens.append(cjk_run[i] + cjk_run[i + 1])
                cjk_run = []
            if ch.isalnum() or ch == "_":
                non_cjk_run.append(ch)
            elif ch == "-":
                # Treat hyphens as word separators (not part of tokens).
                # FTS5 unicode61 tokenizer splits on hyphens during indexing,
                # so queries must also split to avoid "no such column" errors.
                # e.g., "pre-departure" → ["pre", "departure"], not ["pre-departure"]
                if non_cjk_run:
                    word = "".join(non_cjk_run).strip("-")
                    if word:
                        tokens.append(word)
                    non_cjk_run = []
                # Start new token after the hyphen
            else:
                if non_cjk_run:
                    word = "".join(non_cjk_run).strip("-")
                    if word:
                        tokens.append(word)
                    non_cjk_run = []

    # Flush remaining
    if cjk_run:
        for c in cjk_run:
            tokens.append(c)
        for i in range(len(cjk_run) - 1):
            tokens.append(cjk_run[i] + cjk_run[i + 1])
    if non_cjk_run:
        word = "".join(non_cjk_run).strip("-")
        if word:
            tokens.append(word)

    return tokens


# ---------------------------------------------------------------------------
# Basic (non-CJK) tokenizer: word splitting for English etc.
# ---------------------------------------------------------------------------

def _basic_tokenize(text: str) -> list[str]:
    """Tokenize non-CJK text by splitting on word boundaries.

    Extracts alphanumeric sequences (including underscores) as tokens,
    mimicking SQLite FTS5 unicode61 tokenizer behaviour.
    Filters out FTS5 reserved operators.
    """
    tokens = re.findall(r"[\w]+", text)
    return [t for t in tokens if t.upper() not in ("AND", "OR", "NOT", "NEAR")]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _resolve_tokenizer(text: str, tokenizer: str) -> str:
    """Resolve the effective tokenizer based on language detection.

    When *tokenizer* is "auto", detects whether the text contains CJK
    characters and returns:
    - "jieba" if CJK characters are present
    - "basic" (word-level split) otherwise

    For explicit tokenizer values ("jieba", "char"), returns as-is.
    """
    if tokenizer != "auto":
        return tokenizer
    return "jieba" if _has_cjk(text) else "basic"


def tokenize_text(text: str, tokenizer: str = "auto") -> str:
    """Tokenize text for FTS5 indexing.

    Returns space-separated tokens suitable for FTS5 unicode61 tokenizer.
    The FTS5 table indexes the returned string; at query time,
    tokenize_query() produces matching tokens.

    Args:
        text: Raw text content to tokenize.
        tokenizer: "auto" (detect language), "jieba", or "char" (fallback).

    Returns:
        Space-separated token string.
    """
    if not text:
        return ""

    # Resolve "auto" tokenizer based on language detection
    resolved = _resolve_tokenizer(text, tokenizer)

    if resolved == "jieba" and _check_jieba():
        _init_jieba()
        import jieba
        # Use search mode for indexing: produces finer-grained sub-tokens
        # (e.g., "锻炼身体" -> "锻炼", "身体", "锻炼身体")
        # so that queries like "锻炼" can still match.
        words = list(jieba.cut_for_search(text))
        # Filter out whitespace-only and punctuation tokens
        tokens = [w.strip() for w in words if w.strip()]
        return " ".join(tokens)
    elif resolved == "basic":
        tokens = _basic_tokenize(text)
        return " ".join(tokens)
    else:
        # Fallback: character-level + bigram for CJK
        tokens = _fallback_tokenize(text)
        return " ".join(tokens)


def _clean_fts_token(token: str) -> str:
    """Clean a single token for FTS5: remove special chars, keep alphanumeric/CJK."""
    _CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
    parts: list[str] = []
    for ch in token:
        if ch.isalnum() or ch == "_" or _CJK_RE.match(ch):
            parts.append(ch)
    return "".join(parts)


def tokenize_query(text: str, tokenizer: str = "auto") -> list[str]:
    """Tokenize a search query for FTS5 MATCH.

    Returns a list of clean tokens to be joined with spaces
    for the FTS5 MATCH expression.

    Args:
        text: Search query string.
        tokenizer: "auto" (detect language), "jieba", or "char" (fallback).

    Returns:
        List of query tokens.
    """
    if not text:
        return []

    # NFC normalize + full-width conversion
    text = unicodedata.normalize("NFC", text.strip())
    result = []
    for ch in text:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            result.append(" ")
        else:
            result.append(ch)
    text = "".join(result)
    text = re.sub(r"\s+", " ", text)

    # Split hyphenated compound words before tokenization.
    # FTS5's unicode61 tokenizer already treats hyphens as separators during
    # indexing, so queries must also split them to match.
    # Without this, tokens like "pre-departure" cause FTS5 "no such column" errors
    # because FTS5 interprets the hyphen as a column qualifier.
    # Examples: pre-departure → pre departure, cocktail-making → cocktail making
    text = re.sub(r"(\w)-(\w)", r" ", text)

    # Resolve "auto" tokenizer based on language detection
    resolved = _resolve_tokenizer(text, tokenizer)

    if resolved == "jieba" and _check_jieba():
        _init_jieba()
        import jieba
        words = list(jieba.cut(text, cut_all=False))
        tokens = []
        for w in words:
            clean = _clean_fts_token(w.strip())
            if clean and clean.upper() not in ("AND", "OR", "NOT", "NEAR"):
                tokens.append(clean)
        return tokens
    elif resolved == "basic":
        return _basic_tokenize(text)
    else:
        # Fallback: use character-level tokenizer
        tokens = _fallback_tokenize(text)
        return [t for t in tokens if t.upper() not in ("AND", "OR", "NOT", "NEAR")]
