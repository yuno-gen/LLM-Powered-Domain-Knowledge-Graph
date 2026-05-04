"""Tests for :class:`vertexiq.graph_builder.KnowledgeGraph`.

These tests do not call any external service. They verify graph
construction, JSON round-trips, and traversal helpers in isolation.
"""

from __future__ import annotations

import os

import pytest

from vertexiq.graph_builder import KnowledgeGraph


@pytest.fixture
def empty_graph(tmp_path) -> KnowledgeGraph:
    """Return a fresh :class:`KnowledgeGraph` backed by a tmp file."""
    path = tmp_path / "graph.json"
    return KnowledgeGraph(path=str(path))


def test_add_triples_creates_nodes_and_edges(empty_graph: KnowledgeGraph) -> None:
    empty_graph.add_triples([{"subject": "A", "predicate": "leads", "object": "B"}])
    stats = empty_graph.get_stats()
    assert stats["nodes"] == 2
    assert stats["edges"] == 1


def test_add_triples_dedupes_identical_edges(empty_graph: KnowledgeGraph) -> None:
    triple = {"subject": "A", "predicate": "leads", "object": "B"}
    empty_graph.add_triples([triple, triple])
    stats = empty_graph.get_stats()
    assert stats["nodes"] == 2
    assert stats["edges"] == 1


def test_get_neighbors_returns_outgoing_relation(empty_graph: KnowledgeGraph) -> None:
    empty_graph.add_triples([{"subject": "A", "predicate": "leads", "object": "B"}])
    neighbors = empty_graph.get_neighbors("A")
    targets = [n["entity"] for n in neighbors]
    assert "B" in targets
    out_edge = next(n for n in neighbors if n["entity"] == "B")
    assert out_edge["relation"] == "leads"
    assert out_edge["direction"] == "out"


def test_get_neighbors_returns_incoming_relation(empty_graph: KnowledgeGraph) -> None:
    empty_graph.add_triples([{"subject": "A", "predicate": "leads", "object": "B"}])
    neighbors = empty_graph.get_neighbors("B")
    in_edge = next(n for n in neighbors if n["entity"] == "A")
    assert in_edge["direction"] == "in"
    assert in_edge["relation"] == "leads"


def test_get_shortest_path_none_when_target_missing(empty_graph: KnowledgeGraph) -> None:
    empty_graph.add_triples([{"subject": "A", "predicate": "leads", "object": "B"}])
    assert empty_graph.get_shortest_path("A", "C") is None


def test_get_shortest_path_returns_node_list(empty_graph: KnowledgeGraph) -> None:
    empty_graph.add_triples(
        [
            {"subject": "A", "predicate": "knows", "object": "B"},
            {"subject": "B", "predicate": "knows", "object": "C"},
        ]
    )
    path = empty_graph.get_shortest_path("A", "C")
    assert path == ["A", "B", "C"]


def test_save_and_load_roundtrip_preserves_edges(tmp_path) -> None:
    path = tmp_path / "graph.json"
    g1 = KnowledgeGraph(path=str(path))
    g1.add_triples(
        [
            {"subject": "OpenAI", "predicate": "acquired", "object": "Rockset"},
            {"subject": "OpenAI", "predicate": "founded_by", "object": "Sam Altman"},
        ]
    )
    g1.save()
    assert os.path.exists(path)

    g2 = KnowledgeGraph(path=str(path))
    stats = g2.get_stats()
    assert stats["nodes"] == 3
    assert stats["edges"] == 2
    neighbors = {(n["entity"], n["relation"]) for n in g2.get_neighbors("OpenAI")}
    assert ("Rockset", "acquired") in neighbors
    assert ("Sam Altman", "founded_by") in neighbors


def test_entity_exists_is_case_insensitive(empty_graph: KnowledgeGraph) -> None:
    empty_graph.add_triples([{"subject": "OpenAI", "predicate": "acquired", "object": "Rockset"}])
    assert empty_graph.entity_exists("openai") is True
    assert empty_graph.entity_exists("OPENAI") is True
    assert empty_graph.entity_exists("OpenAI") is True
    assert empty_graph.entity_exists("Anthropic") is False


def test_get_ego_graph_respects_depth(empty_graph: KnowledgeGraph) -> None:
    empty_graph.add_triples(
        [
            {"subject": "A", "predicate": "knows", "object": "B"},
            {"subject": "B", "predicate": "knows", "object": "C"},
            {"subject": "C", "predicate": "knows", "object": "D"},
        ]
    )
    one_hop = empty_graph.get_ego_graph("A", depth=1)
    nodes_one = {t["subject"] for t in one_hop} | {t["object"] for t in one_hop}
    assert nodes_one == {"A", "B"}

    two_hop = empty_graph.get_ego_graph("A", depth=2)
    nodes_two = {t["subject"] for t in two_hop} | {t["object"] for t in two_hop}
    assert nodes_two == {"A", "B", "C"}


def test_get_stats_top_entities_by_degree(empty_graph: KnowledgeGraph) -> None:
    empty_graph.add_triples(
        [
            {"subject": "Hub", "predicate": "to", "object": "X"},
            {"subject": "Hub", "predicate": "to", "object": "Y"},
            {"subject": "Hub", "predicate": "to", "object": "Z"},
            {"subject": "X", "predicate": "to", "object": "Y"},
        ]
    )
    stats = empty_graph.get_stats()
    assert stats["top_entities"][0]["entity"] == "Hub"
    assert stats["top_entities"][0]["degree"] == 3
