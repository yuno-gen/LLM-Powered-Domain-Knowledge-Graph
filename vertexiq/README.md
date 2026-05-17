# VertexIQ — LLM-Powered Domain Knowledge Graph Builder

VertexIQ ingests unstructured text from a vertical domain (default: the AI
industry — companies, models, researchers, products, regulations), uses an
LLM with structured outputs to extract `(subject, predicate, object)`
triples, persists them as a directed knowledge graph, and exposes both
graph traversal and natural-language querying via a Streamlit UI and a
FastAPI REST API. Unlike a pure RAG/vector pipeline, VertexIQ keeps the
**explicit relational structure** between entities so multi-hop questions
like *"What connects Microsoft to Rockset?"* resolve via deterministic
graph paths rather than fuzzy similarity search.

---

## Architecture

```
            +-------------------+
            |  Raw documents    |
            |  (text, news,     |
            |   reports, etc.)  |
            +---------+---------+
                      |
                      v
            +-------------------+        OpenAI gpt-4o-mini
            |  extractor.py     |  <-->  (JSON-schema
            |  chunk -> extract |        structured output)
            +---------+---------+
                      | list[{subject, predicate, object}]
                      v
            +-------------------+
            |  graph_builder.py |
            |  NetworkX DiGraph |  <-->  graph.json (persistent)
            +---------+---------+
                      |
       +--------------+--------------+
       |                             |
       v                             v
+--------------+            +--------------------+
| query_engine |            |       UI / API     |
|  structural  |            |  Streamlit  +      |
|  path        |            |  FastAPI    +      |
|  semantic    |            |  pyvis viz         |
|  auto-route  |            +--------------------+
+--------------+
```

- **extractor.py** — chunks the input on sentence boundaries (≈1500 tokens
  per chunk) and calls `gpt-4o-mini` with a strict JSON schema.
- **graph_builder.py** — wraps `networkx.MultiDiGraph`, persists to JSON
  via `node_link_data`, exposes neighbors / shortest path / ego-graph /
  case-insensitive lookup.
- **query_engine.py** — three modes (`structural`, `path`, `semantic`)
  plus an `auto_query` LLM router that picks the right mode and anchor
  entity from a free-form question.
- **api.py** — FastAPI surface (`/extract`, `/stats`, `/neighbors/{e}`,
  `/path`, `/query`, `/auto-query`).
- **app.py** — Streamlit UI: ingest text in the sidebar, view the
  interactive pyvis graph in tab 1, ask questions in tab 2.

---

## Setup

```bash
git clone <your-fork-url> vertexiq && cd vertexiq

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env                # then edit .env and paste your key
# .env contents:
#   OPENAI_API_KEY=sk-...
```

### Run the Streamlit UI

```bash
streamlit run app.py
```

Open http://localhost:8501. The sidebar accepts pasted text; the **Graph
View** tab renders an interactive [pyvis](https://pyvis.readthedocs.io/)
network with nodes shaded darker by degree and edges labeled with their
relation; the **Query** tab answers natural-language questions over the
graph.

### Run the FastAPI server

```bash
uvicorn api:app --reload
```

OpenAPI docs: http://localhost:8000/docs.

### Regenerate the demo graph offline

`graph.json` ships with a 59-node / 64-edge demo graph derived from the
three documents under `sample_docs/`. Regenerate it without spending API
tokens via:

```bash
python -m vertexiq.seed_graph    # from the project root
```

Or, if you do have an API key set, point the API at empty state and
`POST /extract` with the contents of each `sample_docs/*.txt` file.

---

## API reference (curl)

### `POST /extract` — ingest text and persist new triples

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "OpenAI acquired Rockset in 2024 to power ChatGPT retrieval."}'
```

Response:

```json
{ "triples_added": 1, "total_nodes": 60, "total_edges": 65 }
```

### `GET /stats` — graph summary

```bash
curl http://localhost:8000/stats
```

Response:

```json
{
  "nodes": 59,
  "edges": 64,
  "top_entities": [
    {"entity": "OpenAI",   "degree": 15},
    {"entity": "Anthropic","degree": 10}
  ]
}
```

### `GET /neighbors/{entity}` — direct-neighborhood lookup

```bash
curl http://localhost:8000/neighbors/OpenAI
```

Returns 404 if `entity` is not in the graph.

### `GET /path?source=A&target=B` — shortest path

```bash
curl "http://localhost:8000/path?source=Microsoft&target=Rockset"
```

Response:

```json
{ "path": ["Microsoft", "OpenAI", "Rockset"], "hops": 2, "exists": true }
```

### `POST /query` — semantic, LLM-grounded answer

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Who did OpenAI acquire?", "anchor_entity": "OpenAI"}'
```

The LLM is given **only** the 2-hop ego graph of the anchor as context.

### `POST /auto-query` — let VertexIQ pick the mode

```bash
curl -X POST http://localhost:8000/auto-query \
  -H "Content-Type: application/json" \
  -d '{"question": "How is Microsoft connected to Rockset?"}'
```

The router picks `structural`, `path`, or `semantic` and an anchor
entity, then dispatches to the matching engine method.

---


## Tests

```bash
pytest tests/ -v
```

The suite (28 tests) is fully offline — the OpenAI client is stubbed in
the extractor and query-engine tests so no API key is required to run
CI.

---

## Why Knowledge Graphs over Pure RAG

Vector search retrieves passages that are **semantically similar** to a
query, which makes it strong for surface-level lookups but unreliable
when the question requires composing multiple discrete facts. A
knowledge graph stores entities and relations as first-class objects,
so multi-hop reasoning (`A` is acquired by `B`, `B` is regulated by
`C`, therefore `A`'s parent now sits under regulation `C`) becomes a
deterministic path traversal rather than a similarity heuristic. This
also makes provenance auditable — every fact in an answer maps back to
a specific edge — and lets you constrain LLM context to *only* the
anchor entity's ego graph, slashing hallucination risk versus stuffing
top-k chunks into the prompt.
