"""FastAPI REST surface for VertexIQ.

Run with:

    uvicorn vertexiq.api:app --reload

The graph file path defaults to ``graph.json`` in the current working
directory and can be overridden via the ``VERTEXIQ_GRAPH_PATH``
environment variable. A single :class:`KnowledgeGraph` instance is
shared across requests; mutations are persisted to disk on each
``/extract`` call.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Support both ``uvicorn vertexiq.api:app`` (package import) and
# ``uvicorn api:app`` from inside the ``vertexiq/`` directory.
try:  # pragma: no cover - import path resolution
    from .extractor import extract_triples
    from .graph_builder import KnowledgeGraph
    from .query_engine import QueryEngine
except ImportError:  # pragma: no cover
    from extractor import extract_triples  # type: ignore[no-redef]
    from graph_builder import KnowledgeGraph  # type: ignore[no-redef]
    from query_engine import QueryEngine  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

GRAPH_PATH = os.getenv("VERTEXIQ_GRAPH_PATH", "graph.json")

app = FastAPI(
    title="VertexIQ",
    description="LLM-Powered Domain Knowledge Graph Builder",
    version="0.1.0",
)

# Shared, request-scoped helpers. Created lazily so importing this
# module never touches OpenAI; useful for tooling like `--help`.
_graph: KnowledgeGraph | None = None


def get_graph() -> KnowledgeGraph:
    """Return the process-wide :class:`KnowledgeGraph` singleton."""
    global _graph
    if _graph is None:
        _graph = KnowledgeGraph(path=GRAPH_PATH)
    return _graph


def get_engine() -> QueryEngine:
    """Return a :class:`QueryEngine` bound to the singleton graph."""
    return QueryEngine(get_graph())


# ----------------------------------------------------------------------
# Request / response models
# ----------------------------------------------------------------------
class ExtractRequest(BaseModel):
    """Body for ``POST /extract``."""

    text: str = Field(..., min_length=1, description="Document text to extract from.")


class ExtractResponse(BaseModel):
    """Response for ``POST /extract``."""

    triples_added: int
    total_nodes: int
    total_edges: int


class QueryRequest(BaseModel):
    """Body for ``POST /query``."""

    question: str = Field(..., min_length=1)
    anchor_entity: str = Field(..., min_length=1)


class AutoQueryRequest(BaseModel):
    """Body for ``POST /auto-query``."""

    question: str = Field(..., min_length=1)


class AnswerResponse(BaseModel):
    """Generic ``{"answer": str}`` payload."""

    answer: str


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest) -> ExtractResponse:
    """Run extraction on ``text`` and persist the resulting triples."""
    try:
        triples = extract_triples(req.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive net
        logger.exception("Extraction failed")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc

    graph = get_graph()
    added = graph.add_triples(triples)
    graph.save()
    stats = graph.get_stats()
    return ExtractResponse(
        triples_added=added,
        total_nodes=stats["nodes"],
        total_edges=stats["edges"],
    )


@app.get("/stats")
def stats() -> dict:
    """Return graph stats (node/edge counts and top entities)."""
    return get_graph().get_stats()


@app.get("/neighbors/{entity}")
def neighbors(entity: str) -> dict:
    """Return all neighbors of ``entity`` (404 if missing)."""
    graph = get_graph()
    if not graph.entity_exists(entity):
        raise HTTPException(status_code=404, detail=f"Entity '{entity}' not found")
    resolved = graph._resolve(entity)
    return {"entity": resolved, "neighbors": graph.get_neighbors(entity)}


@app.get("/path")
def path(source: str, target: str) -> dict:
    """Compute shortest path between two entities."""
    engine = get_engine()
    return engine.path_query(source, target)


@app.post("/query", response_model=AnswerResponse)
def query(req: QueryRequest) -> AnswerResponse:
    """Run a semantic, LLM-grounded query anchored on a specific entity."""
    engine = get_engine()
    try:
        answer = engine.semantic_query(req.question, req.anchor_entity)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
    return AnswerResponse(answer=answer)


@app.post("/auto-query", response_model=AnswerResponse)
def auto_query(req: AutoQueryRequest) -> AnswerResponse:
    """Route the question to the best mode and return a formatted answer."""
    engine = get_engine()
    try:
        answer = engine.auto_query(req.question)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Auto-query failed")
        raise HTTPException(status_code=500, detail=f"Auto-query failed: {exc}") from exc
    return AnswerResponse(answer=answer)
