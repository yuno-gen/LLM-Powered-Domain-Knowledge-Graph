"""Persistent NetworkX-backed knowledge graph.

The :class:`KnowledgeGraph` wraps :class:`networkx.DiGraph` and adds:

    - JSON serialization to a file path (default ``graph.json``).
    - Triple ingestion that stores the predicate as the edge attribute
      ``relation``.
    - Convenience traversals used by the query engine and the API
      (neighbors, shortest path, ego graph).
    - Case-insensitive entity lookup so callers don't have to track the
      exact casing the LLM produced.

JSON layout matches :func:`networkx.node_link_data` with
``edges="links"`` so the file can be round-tripped without warnings on
modern NetworkX versions.
"""

from __future__ import annotations

import json
import os
from typing import Any

import networkx as nx

try:  # pragma: no cover - import path resolution
    from .utils import normalize_entity
except ImportError:  # pragma: no cover
    from utils import normalize_entity  # type: ignore[no-redef]

_NODE_LINK_KWARGS: dict[str, Any] = {"edges": "links"}


class KnowledgeGraph:
    """A persistent directed knowledge graph keyed by entity name.

    Edges carry a single ``relation`` attribute (the predicate). Parallel
    edges with different relations between the same node pair are
    supported via :class:`networkx.MultiDiGraph` semantics, but the
    default backing store is :class:`networkx.DiGraph`. Adding a
    duplicate ``(subject, predicate, object)`` is idempotent.
    """

    def __init__(self, path: str = "graph.json") -> None:
        """Initialize an empty graph and try to load from ``path``.

        Args:
            path: File path used for both :meth:`save` and :meth:`load`.
                If the file does not exist, the graph stays empty.
        """
        self.path = path
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        if os.path.exists(path):
            try:
                self.load()
            except Exception:
                # A corrupt graph file should not prevent the app from
                # starting; the user can rebuild via /extract.
                self.graph = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add_triples(self, triples: list[dict]) -> int:
        """Add triples as directed edges with a ``relation`` attribute.

        Args:
            triples: Sequence of ``{subject, predicate, object}`` dicts.

        Returns:
            Number of edges actually added (deduplicated against
            existing ``(subject, relation, object)`` tuples).
        """
        added = 0
        for t in triples or []:
            subject = normalize_entity(str(t.get("subject", "")))
            predicate = str(t.get("predicate", "")).strip()
            obj = normalize_entity(str(t.get("object", "")))
            if not (subject and predicate and obj):
                continue
            if self._edge_exists(subject, obj, predicate):
                continue
            self.graph.add_node(subject)
            self.graph.add_node(obj)
            self.graph.add_edge(subject, obj, relation=predicate)
            added += 1
        return added

    def _edge_exists(self, source: str, target: str, relation: str) -> bool:
        """Return True if an edge ``source -[relation]-> target`` exists."""
        if not self.graph.has_edge(source, target):
            return False
        for data in self.graph[source][target].values():
            if data.get("relation") == relation:
                return True
        return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        """Serialize the graph to ``self.path`` as JSON."""
        data = nx.node_link_data(self.graph, **_NODE_LINK_KWARGS)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self) -> None:
        """Deserialize the graph from ``self.path``."""
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data, directed=True, multigraph=True, **_NODE_LINK_KWARGS)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        """Summary statistics for the graph.

        Returns:
            ``{"nodes": int, "edges": int, "top_entities": list[dict]}``
            where ``top_entities`` is the five highest-degree nodes,
            each as ``{"entity": str, "degree": int}``.
        """
        degrees = sorted(self.graph.degree(), key=lambda x: x[1], reverse=True)
        top = [{"entity": n, "degree": d} for n, d in degrees[:5]]
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "top_entities": top,
        }

    def _resolve(self, entity: str) -> str | None:
        """Return the actual node name matching ``entity`` case-insensitively."""
        if not entity:
            return None
        if self.graph.has_node(entity):
            return entity
        target = entity.lower()
        for node in self.graph.nodes:
            if node.lower() == target:
                return node
        return None

    def entity_exists(self, entity: str) -> bool:
        """Case-insensitive membership check."""
        return self._resolve(entity) is not None

    def get_neighbors(self, entity: str) -> list[dict]:
        """Return all neighbors of ``entity`` with relation + direction.

        Args:
            entity: Entity name (case-insensitive).

        Returns:
            List of ``{"entity": str, "relation": str, "direction":
            "out"|"in"}``. Empty list if ``entity`` is not in the graph.
        """
        node = self._resolve(entity)
        if node is None:
            return []
        out: list[dict] = []
        for _, target, data in self.graph.out_edges(node, data=True):
            out.append({"entity": target, "relation": data.get("relation", ""), "direction": "out"})
        for source, _, data in self.graph.in_edges(node, data=True):
            out.append({"entity": source, "relation": data.get("relation", ""), "direction": "in"})
        return out

    def get_shortest_path(self, source: str, target: str) -> list[str] | None:
        """Return the shortest undirected path between two entities.

        The path is computed on the undirected projection so that
        relations like ``acquired`` (one-way) still surface
        relationships when the user asks "what connects X and Y".

        Args:
            source: Start entity.
            target: End entity.

        Returns:
            Ordered list of node names, or ``None`` if either entity is
            missing or no path exists.
        """
        s = self._resolve(source)
        t = self._resolve(target)
        if s is None or t is None:
            return None
        try:
            undirected = self.graph.to_undirected(as_view=False)
            return nx.shortest_path(undirected, source=s, target=t)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_ego_graph(self, entity: str, depth: int = 2) -> list[dict]:
        """Return all triples within ``depth`` hops of ``entity``.

        The traversal respects the underlying directed structure but
        crosses both in- and out-edges so that important context like
        "Acquirer-of" is included for the queried entity.

        Args:
            entity: Anchor entity (case-insensitive).
            depth: Maximum hops from the anchor.

        Returns:
            List of ``{"subject", "predicate", "object"}`` dicts.
            Empty list if ``entity`` is not in the graph.
        """
        node = self._resolve(entity)
        if node is None:
            return []
        undirected = self.graph.to_undirected(as_view=False)
        ego_nodes: set[str] = set(
            nx.single_source_shortest_path_length(undirected, node, cutoff=depth).keys()
        )
        triples: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for u, v, data in self.graph.edges(data=True):
            if u in ego_nodes and v in ego_nodes:
                relation = data.get("relation", "")
                key = (u, relation, v)
                if key in seen:
                    continue
                seen.add(key)
                triples.append({"subject": u, "predicate": relation, "object": v})
        return triples
