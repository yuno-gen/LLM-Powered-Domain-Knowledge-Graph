"""Tests for :class:`vertexiq.query_engine.QueryEngine`.

Network-bound paths (semantic_query, auto_query) are exercised with a
stub OpenAI client so the suite remains fully offline.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from vertexiq import query_engine
from vertexiq.graph_builder import KnowledgeGraph
from vertexiq.query_engine import QueryEngine


def _make_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class _StubClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

        outer = self

        class _Completions:
            @staticmethod
            def create(**kwargs):
                outer.calls.append(kwargs)
                if not outer._responses:
                    return _make_response("(empty)")
                return _make_response(outer._responses.pop(0))

        self.chat = SimpleNamespace(completions=_Completions())


@pytest.fixture
def populated_graph(tmp_path) -> KnowledgeGraph:
    """Graph populated with a small AI-industry sample."""
    g = KnowledgeGraph(path=str(tmp_path / "graph.json"))
    g.add_triples(
        [
            {"subject": "OpenAI", "predicate": "acquired", "object": "Rockset"},
            {"subject": "OpenAI", "predicate": "founded_by", "object": "Sam Altman"},
            {"subject": "Microsoft", "predicate": "invested_in", "object": "OpenAI"},
            {"subject": "Anthropic", "predicate": "founded_by", "object": "Dario Amodei"},
        ]
    )
    return g


def test_structural_query_returns_neighbors(populated_graph: KnowledgeGraph) -> None:
    engine = QueryEngine(populated_graph)
    result = engine.structural_query("OpenAI")
    assert result["entity"] == "OpenAI"
    assert result["degree"] >= 1
    assert len(result["neighbors"]) > 0
    assert "Rockset" in result["connected_to"]
    assert "Sam Altman" in result["connected_to"]


def test_structural_query_is_case_insensitive(populated_graph: KnowledgeGraph) -> None:
    engine = QueryEngine(populated_graph)
    result = engine.structural_query("openai")
    assert result["entity"] == "OpenAI"


def test_structural_query_raises_for_missing_entity(populated_graph: KnowledgeGraph) -> None:
    engine = QueryEngine(populated_graph)
    with pytest.raises(ValueError, match="not found"):
        engine.structural_query("NonExistentEntity")


def test_path_query_connected(populated_graph: KnowledgeGraph) -> None:
    engine = QueryEngine(populated_graph)
    result = engine.path_query("Microsoft", "Rockset")
    assert result["exists"] is True
    assert result["hops"] is not None and result["hops"] >= 1
    assert result["path"] is not None


def test_path_query_disconnected(populated_graph: KnowledgeGraph) -> None:
    engine = QueryEngine(populated_graph)
    result = engine.path_query("OpenAI", "Anthropic")
    assert result["exists"] is False
    assert result["path"] is None
    assert result["hops"] is None


def test_path_query_missing_endpoint(populated_graph: KnowledgeGraph) -> None:
    engine = QueryEngine(populated_graph)
    result = engine.path_query("OpenAI", "Ghost")
    assert result["exists"] is False


def test_semantic_query_returns_non_empty_string(
    populated_graph: KnowledgeGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _StubClient(["OpenAI acquired Rockset in 2024."])
    monkeypatch.setattr(query_engine, "_get_client", lambda: stub)

    engine = QueryEngine(populated_graph)
    answer = engine.semantic_query("Who did OpenAI acquire?", "OpenAI")
    assert isinstance(answer, str)
    assert len(answer) > 0
    # Verify the graph context was actually included in the prompt.
    user_msg = stub.calls[0]["messages"][1]["content"]
    assert "OpenAI" in user_msg
    assert "Rockset" in user_msg


def test_semantic_query_raises_for_missing_anchor(
    populated_graph: KnowledgeGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(query_engine, "_get_client", lambda: _StubClient([]))
    engine = QueryEngine(populated_graph)
    with pytest.raises(ValueError, match="not found"):
        engine.semantic_query("anything?", "Ghost")


def test_semantic_query_raises_for_empty_question(
    populated_graph: KnowledgeGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(query_engine, "_get_client", lambda: _StubClient([]))
    engine = QueryEngine(populated_graph)
    with pytest.raises(ValueError, match="non-empty"):
        engine.semantic_query("   ", "OpenAI")


def test_auto_query_routes_to_structural(
    populated_graph: KnowledgeGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    routing = json.dumps({"mode": "structural", "anchor": "OpenAI", "target": None})
    stub = _StubClient([routing])
    monkeypatch.setattr(query_engine, "_get_client", lambda: stub)

    engine = QueryEngine(populated_graph)
    result = engine.auto_query("What is OpenAI connected to?")
    assert isinstance(result, str)
    assert "OpenAI" in result
    assert "Rockset" in result or "Sam Altman" in result


def test_auto_query_routes_to_path(
    populated_graph: KnowledgeGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    routing = json.dumps({"mode": "path", "anchor": "Microsoft", "target": "Rockset"})
    stub = _StubClient([routing])
    monkeypatch.setattr(query_engine, "_get_client", lambda: stub)

    engine = QueryEngine(populated_graph)
    result = engine.auto_query("How is Microsoft connected to Rockset?")
    assert "Path" in result
    assert "Microsoft" in result


def test_auto_query_raises_on_empty_graph(tmp_path) -> None:
    g = KnowledgeGraph(path=str(tmp_path / "graph.json"))
    engine = QueryEngine(g)
    with pytest.raises(ValueError, match="empty"):
        engine.auto_query("Anything?")
