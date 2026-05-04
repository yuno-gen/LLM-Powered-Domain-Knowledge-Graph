"""Triple extraction via OpenAI structured outputs.

The extractor wraps the OpenAI Chat Completions API with a JSON-schema
constraint so the model is forced to return a list of
``{subject, predicate, object}`` objects. Long inputs are chunked via
:func:`vertexiq.utils.chunk_text` before being sent to the model.

Design notes:
    - Failure modes are explicit: empty text → ``[]``, malformed JSON →
      log and return ``[]`` for that chunk, transient API errors → one
      retry then re-raise.
    - The OpenAI client is created lazily so that importing this module
      never crashes when ``OPENAI_API_KEY`` is unset (important for
      unit tests that don't touch the network).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from dotenv import load_dotenv

try:  # pragma: no cover - import path resolution
    from .utils import chunk_text, normalize_entity
except ImportError:  # pragma: no cover
    from utils import chunk_text, normalize_entity  # type: ignore[no-redef]

load_dotenv()

logger = logging.getLogger(__name__)

_MODEL = "gpt-4o-mini"

_SYSTEM_PROMPT = (
    "You are a knowledge graph extraction engine. Extract all meaningful "
    "entity-relationship triples from the text. Focus on named entities "
    "(people, companies, products, technologies, regulations). Return ONLY "
    "a JSON array of objects with keys: subject (str), predicate (str), "
    "object (str). Normalize entity names to title case. Use concise, "
    "consistent predicate verbs (e.g., 'acquired', 'founded', "
    "'partnered_with', 'released', 'regulates')."
)

# JSON schema enforced via response_format. Wrapped in an object because
# the OpenAI structured-output schema must be an object at the top level.
_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                },
                "required": ["subject", "predicate", "object"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["triples"],
    "additionalProperties": False,
}


def _get_client():
    """Lazily build and return an OpenAI client.

    Raises:
        RuntimeError: if ``OPENAI_API_KEY`` is missing from the
        environment.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add "
            "your key, or export it in the shell before calling the "
            "extractor."
        )
    from openai import OpenAI  # imported lazily so tests don't require the SDK

    return OpenAI(api_key=api_key)


def _call_model(client, chunk: str) -> str:
    """Invoke the chat completion endpoint for a single chunk.

    Returns the raw assistant content string. The caller is responsible
    for parsing JSON.
    """
    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": chunk},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "triples_response",
                "schema": _RESPONSE_SCHEMA,
                "strict": True,
            },
        },
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


def _extract_chunk(client, chunk: str) -> list[dict]:
    """Extract triples from a single chunk with one retry on transient errors."""
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            raw = _call_model(client, chunk)
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning("Malformed JSON from model (attempt %d): %s", attempt + 1, exc)
                return []
            triples = payload.get("triples", []) if isinstance(payload, dict) else []
            return _clean_triples(triples)
        except Exception as exc:  # pragma: no cover - defensive net for SDK errors
            last_error = exc
            logger.warning("OpenAI call failed (attempt %d/2): %s", attempt + 1, exc)
            if attempt == 0:
                time.sleep(1)
                continue
            raise
    # Unreachable but keeps type checkers happy.
    if last_error:  # pragma: no cover
        raise last_error
    return []


def _clean_triples(raw: list[Any]) -> list[dict]:
    """Validate and normalize raw triple dicts coming from the model."""
    cleaned: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        subject = normalize_entity(str(item.get("subject", "")))
        predicate = str(item.get("predicate", "")).strip().lower().replace(" ", "_")
        obj = normalize_entity(str(item.get("object", "")))
        if not (subject and predicate and obj):
            continue
        if subject == obj:
            continue
        key = (subject, predicate, obj)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"subject": subject, "predicate": predicate, "object": obj})
    return cleaned


def extract_triples(text: str) -> list[dict]:
    """Extract subject-predicate-object triples from ``text``.

    The function chunks long inputs (≈1500 tokens per chunk), calls the
    OpenAI model with a strict JSON schema, deduplicates triples across
    chunks and returns a flat list. The function is total: it always
    returns a list, never ``None``.

    Args:
        text: Source document text.

    Returns:
        List of dicts shaped ``{"subject": str, "predicate": str,
        "object": str}``. Empty list for empty input or when the model
        produces no usable triples.

    Raises:
        RuntimeError: if ``OPENAI_API_KEY`` is unset.
        Exception: re-raised after a single retry on persistent API
            failures.
    """
    if not text or not text.strip():
        return []

    chunks = chunk_text(text, max_tokens=1500)
    if not chunks:
        return []

    client = _get_client()

    aggregated: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for chunk in chunks:
        for triple in _extract_chunk(client, chunk):
            key = (triple["subject"], triple["predicate"], triple["object"])
            if key in seen:
                continue
            seen.add(key)
            aggregated.append(triple)
    return aggregated
