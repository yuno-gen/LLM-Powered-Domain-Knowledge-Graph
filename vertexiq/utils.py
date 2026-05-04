"""Shared helpers for VertexIQ.

This module contains pure-Python utilities with no external service
dependencies. Anything in here must remain side-effect free so that the
rest of the package (extractor, graph_builder, query_engine) can rely
on it without setup overhead in tests.
"""

from __future__ import annotations

import re
from typing import Iterable

# Approximate characters-per-token ratio used for cheap, deterministic
# token estimation. The real tokenizer is intentionally not invoked here
# because we want chunking to work in environments where tiktoken is
# unavailable and the OpenAI client may not be installed yet.
_CHARS_PER_TOKEN = 4


def _approx_token_count(text: str) -> int:
    """Approximate the token count of ``text`` using a 4-chars-per-token heuristic."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences on ``.``, ``!``, ``?`` boundaries.

    Falls back to a single-element list if no terminators are found so
    that callers always receive at least one chunk for non-empty input.
    """
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, max_tokens: int = 1500) -> list[str]:
    """Split ``text`` into chunks no larger than ``max_tokens`` (approx).

    Sentences are kept intact whenever possible. A single sentence that
    on its own exceeds ``max_tokens`` is hard-wrapped on character
    boundaries so that no chunk silently exceeds the model context.

    Args:
        text: Input text to chunk.
        max_tokens: Approximate maximum tokens per chunk (default 1500).

    Returns:
        List of chunk strings. Returns ``[]`` if ``text`` is empty or
        whitespace-only.
    """
    if not text or not text.strip():
        return []

    max_chars = max_tokens * _CHARS_PER_TOKEN
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    for sentence in sentences:
        sentence_len = len(sentence) + 1  # +1 for joining space
        if sentence_len > max_chars:
            # Flush current buffer first.
            if buffer:
                chunks.append(" ".join(buffer).strip())
                buffer, buffer_len = [], 0
            # Hard-wrap oversized sentence on character boundaries.
            for start in range(0, len(sentence), max_chars):
                chunks.append(sentence[start : start + max_chars])
            continue

        if buffer_len + sentence_len > max_chars and buffer:
            chunks.append(" ".join(buffer).strip())
            buffer, buffer_len = [], 0

        buffer.append(sentence)
        buffer_len += sentence_len

    if buffer:
        chunks.append(" ".join(buffer).strip())

    return chunks


def format_triples(triples: Iterable[dict]) -> str:
    """Format an iterable of triples into a human-readable string.

    Each triple is rendered as ``Subject [predicate] Object`` on its
    own line. Triples missing any of the three required keys are
    skipped silently because incomplete data is a routine outcome from
    LLM extraction and should not break presentation.

    Args:
        triples: Iterable of dicts with ``subject``, ``predicate``,
            ``object`` keys.

    Returns:
        A newline-joined string. Empty string when no valid triples.
    """
    lines: list[str] = []
    for t in triples:
        subj = t.get("subject")
        pred = t.get("predicate")
        obj = t.get("object")
        if subj and pred and obj:
            lines.append(f"{subj} [{pred}] {obj}")
    return "\n".join(lines)


def normalize_entity(name: str) -> str:
    """Normalize an entity name to title case with collapsed whitespace.

    Empty or whitespace-only input returns ``""`` so callers can use
    a falsy check to skip junk entities.

    Args:
        name: Raw entity string.

    Returns:
        Cleaned, title-cased string.
    """
    if not name:
        return ""
    cleaned = re.sub(r"\s+", " ", name).strip()
    if not cleaned:
        return ""
    # Title-case while preserving common all-caps acronyms (OpenAI-style
    # names with internal capitals are returned by the LLM already and
    # are left alone if they contain mixed case).
    if any(c.isupper() for c in cleaned[1:]):
        return cleaned
    return cleaned.title()
