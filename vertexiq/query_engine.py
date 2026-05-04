"""Query layer over a :class:`~vertexiq.graph_builder.KnowledgeGraph`.

Exposes three query modes:

    1. :meth:`QueryEngine.structural_query` — direct neighborhood/degree
       lookup. Pure graph operation, no LLM call.
    2. :meth:`QueryEngine.path_query` — shortest path between two
       entities. Pure graph operation.
    3. :meth:`QueryEngine.semantic_query` — LLM-grounded answer using
       only the triples in the entity's 2-hop ego graph.

The :meth:`QueryEngine.auto_query` method dispatches to one of the
above based on an LLM classification step, so non-technical users can
ask questions in plain English without picking a mode.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv

try:  # pragma: no cover - import path resolution
    from .graph_builder import KnowledgeGraph
    from .utils import format_triples
except ImportError:  # pragma: no cover
    from graph_builder import KnowledgeGraph  # type: ignore[no-redef]
    from utils import format_triples  # type: ignore[no-redef]

load_dotenv()

logger = logging.getLogger(__name__)

_MODEL = "gpt-4o-mini"

_SEMANTIC_SYSTEM_PROMPT = (
    "You are a knowledge graph analyst. Answer the user's question using "
    "ONLY the provided graph triples as context. If the answer is not in "
    "the graph, say so explicitly. Do not hallucinate relationships."
)

_ROUTER_SYSTEM_PROMPT = (
    "You are a query router for a knowledge graph. Given a user question "
    "and a list of entities present in the graph, decide:\n"
    "  - mode: one of 'structural', 'path', 'semantic'.\n"
    "    * 'structural' for questions about direct relationships of one "
    "entity (e.g. 'who works for X', 'what does X do').\n"
    "    * 'path' for questions about how two entities are connected.\n"
    "    * 'semantic' for open-ended or multi-hop questions.\n"
    "  - anchor: the primary entity from the graph (must be one of the "
    "given entities, exact spelling).\n"
    "  - target: only for path mode, the second entity (must be in the "
    "given entities); otherwise null.\n"
    "Return strict JSON with keys 'mode', 'anchor', 'target'."
)


def _get_client():
    """Lazily build an OpenAI client; raise if no API key configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. The query engine needs it for "
            "semantic_query / auto_query."
        )
    from openai import OpenAI

    return OpenAI(api_key=api_key)


class QueryEngine:
    """High-level query interface over a :class:`KnowledgeGraph`."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        """Bind the engine to a graph instance.

        Args:
            graph: Already-constructed :class:`KnowledgeGraph` (loaded
                or freshly built). The engine never mutates it.
        """
        self.graph = graph

    # ------------------------------------------------------------------
    # Structural / Path (no LLM)
    # ------------------------------------------------------------------
    def structural_query(self, entity: str) -> dict:
        """Return direct-neighborhood facts about ``entity``.

        Args:
            entity: Entity name (case-insensitive).

        Returns:
            ``{"entity": resolved_name, "neighbors": list[dict],
            "degree": int, "connected_to": list[str]}``.

        Raises:
            ValueError: if ``entity`` is not in the graph.
        """
        resolved = self.graph._resolve(entity)
        if resolved is None:
            raise ValueError(
                f"Entity '{entity}' not found in the graph. "
                "Run /extract first or check the spelling."
            )
        neighbors = self.graph.get_neighbors(resolved)
        connected_to = sorted({n["entity"] for n in neighbors})
        return {
            "entity": resolved,
            "neighbors": neighbors,
            "degree": self.graph.graph.degree(resolved),
            "connected_to": connected_to,
        }

    def path_query(self, source: str, target: str) -> dict:
        """Compute shortest path between two entities.

        Args:
            source: Start entity (case-insensitive).
            target: End entity (case-insensitive).

        Returns:
            ``{"path": list[str] | None, "hops": int | None,
            "exists": bool}``. ``hops`` is ``len(path) - 1``.
        """
        path = self.graph.get_shortest_path(source, target)
        if path is None:
            return {"path": None, "hops": None, "exists": False}
        return {"path": path, "hops": max(len(path) - 1, 0), "exists": True}

    # ------------------------------------------------------------------
    # Semantic (LLM-grounded)
    # ------------------------------------------------------------------
    def semantic_query(self, question: str, anchor_entity: str) -> str:
        """Answer ``question`` using triples within 2 hops of ``anchor_entity``.

        Args:
            question: Free-form natural language question.
            anchor_entity: Entity in the graph used to scope context.

        Returns:
            Model-generated answer string.

        Raises:
            ValueError: if ``anchor_entity`` is not in the graph or the
                question is empty.
        """
        if not question or not question.strip():
            raise ValueError("Question must be non-empty.")
        if not self.graph.entity_exists(anchor_entity):
            raise ValueError(
                f"Anchor entity '{anchor_entity}' not found in the graph."
            )

        triples = self.graph.get_ego_graph(anchor_entity, depth=2)
        context = format_triples(triples) or "(no triples found within 2 hops)"

        client = _get_client()
        user_content = (
            f"Graph context (triples within 2 hops of '{anchor_entity}'):\n"
            f"{context}\n\n"
            f"Question: {question}"
        )
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SEMANTIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
        )
        answer = response.choices[0].message.content or ""
        return answer.strip()

    # ------------------------------------------------------------------
    # Auto-routed
    # ------------------------------------------------------------------
    def auto_query(self, question: str) -> str:
        """Route ``question`` to the best mode and return a formatted answer.

        Uses the LLM as a classifier to pick mode + anchor (and target
        for path mode). Falls back to semantic mode anchored on the
        highest-degree entity if classification fails or the chosen
        anchor isn't in the graph.

        Args:
            question: Free-form natural language question.

        Returns:
            Formatted, human-readable answer string.

        Raises:
            ValueError: if the graph is empty.
        """
        if not question or not question.strip():
            raise ValueError("Question must be non-empty.")
        entities = list(self.graph.graph.nodes)
        if not entities:
            raise ValueError("Graph is empty; ingest documents via /extract first.")

        decision = self._route(question, entities)
        mode = decision.get("mode", "semantic")
        anchor = decision.get("anchor")
        target = decision.get("target")

        if not anchor or not self.graph.entity_exists(anchor):
            anchor = self._fallback_anchor(entities)

        if mode == "structural":
            try:
                result = self.structural_query(anchor)
            except ValueError as exc:
                return str(exc)
            return self._format_structural(result)
        if mode == "path" and target and self.graph.entity_exists(target):
            result = self.path_query(anchor, target)
            return self._format_path(anchor, target, result)
        # default: semantic
        return self.semantic_query(question, anchor)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _route(self, question: str, entities: list[str]) -> dict[str, Any]:
        """Ask the LLM to classify the query mode and pick anchors."""
        try:
            client = _get_client()
        except RuntimeError:
            return {"mode": "structural", "anchor": None, "target": None}

        # Cap entity list to avoid blowing the context window on large graphs.
        sample = entities[:200]
        user_content = (
            f"Entities in graph: {json.dumps(sample)}\n"
            f"Question: {question}\n\n"
            "Respond with JSON only."
        )
        try:
            response = client.chat.completions.create(
                model=_MODEL,
                messages=[
                    {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            raw = response.choices[0].message.content or "{}"
            return json.loads(raw)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("auto_query routing failed: %s", exc)
            return {"mode": "semantic", "anchor": None, "target": None}

    def _fallback_anchor(self, entities: list[str]) -> str:
        """Pick the highest-degree node as a safe default anchor."""
        degrees = sorted(self.graph.graph.degree(), key=lambda x: x[1], reverse=True)
        if degrees:
            return degrees[0][0]
        return entities[0]

    @staticmethod
    def _format_structural(result: dict) -> str:
        lines = [f"Entity: {result['entity']} (degree {result['degree']})"]
        if not result["neighbors"]:
            lines.append("  (no relationships)")
        else:
            for n in result["neighbors"]:
                arrow = "->" if n["direction"] == "out" else "<-"
                lines.append(f"  {result['entity']} {arrow} [{n['relation']}] {n['entity']}")
        return "\n".join(lines)

    @staticmethod
    def _format_path(source: str, target: str, result: dict) -> str:
        if not result["exists"]:
            return f"No path between {source!r} and {target!r}."
        path = " -> ".join(result["path"])
        return f"Path ({result['hops']} hops): {path}"
