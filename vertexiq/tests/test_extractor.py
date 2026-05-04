"""Tests for :mod:`vertexiq.extractor`.

Network calls are stubbed via :func:`monkeypatch` so the suite runs
deterministically and offline. The shape of the stub mirrors the
``openai`` SDK: ``client.chat.completions.create`` returns an object
whose ``.choices[0].message.content`` is a JSON string matching the
schema the extractor enforces.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from vertexiq import extractor


def _make_response(content: str):
    """Construct a minimal mock that quacks like an OpenAI ChatCompletion."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class _StubClient:
    """Stub OpenAI client that returns a queued sequence of responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

        outer = self

        class _Completions:
            @staticmethod
            def create(**kwargs):  # noqa: D401 - mirrors SDK signature
                outer.calls.append(kwargs)
                if not outer._responses:
                    return _make_response('{"triples": []}')
                return _make_response(outer._responses.pop(0))

        self.chat = SimpleNamespace(completions=_Completions())


@pytest.fixture
def patch_client(monkeypatch: pytest.MonkeyPatch):
    """Patch :func:`extractor._get_client` with a controllable stub."""

    def _factory(responses: list[str]) -> _StubClient:
        stub = _StubClient(responses)
        monkeypatch.setattr(extractor, "_get_client", lambda: stub)
        return stub

    return _factory


def test_returns_list_for_simple_acquisition(patch_client) -> None:
    payload = {
        "triples": [
            {"subject": "OpenAI", "predicate": "acquired", "object": "Rockset"}
        ]
    }
    patch_client([json.dumps(payload)])

    triples = extractor.extract_triples("OpenAI acquired Rockset in 2024.")

    assert isinstance(triples, list)
    assert len(triples) >= 1
    first = triples[0]
    assert first["subject"] == "OpenAI"
    assert "acquired" in first["predicate"]
    assert first["object"] == "Rockset"


def test_empty_string_returns_empty_list(patch_client) -> None:
    patch_client([])
    assert extractor.extract_triples("") == []


def test_whitespace_only_returns_empty_list(patch_client) -> None:
    patch_client([])
    assert extractor.extract_triples("   \n\t  ") == []


def test_no_named_entities_returns_empty_list(patch_client) -> None:
    patch_client(['{"triples": []}'])
    triples = extractor.extract_triples("It was a quiet, uneventful afternoon.")
    assert isinstance(triples, list)
    assert triples == []


def test_output_is_always_a_list_even_on_malformed_json(patch_client) -> None:
    patch_client(["this is not json"])
    triples = extractor.extract_triples("Acme acquired Beta.")
    assert isinstance(triples, list)
    assert triples == []


def test_dedupes_across_chunks(patch_client) -> None:
    duplicate = json.dumps(
        {
            "triples": [
                {"subject": "OpenAI", "predicate": "acquired", "object": "Rockset"},
                {"subject": "OpenAI", "predicate": "acquired", "object": "Rockset"},
            ]
        }
    )
    patch_client([duplicate])

    triples = extractor.extract_triples("OpenAI acquired Rockset.")
    assert len(triples) == 1
