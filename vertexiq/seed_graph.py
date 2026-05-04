"""Seed a demo ``graph.json`` from hand-curated AI-industry triples.

This script is a convenience for offline demos. It produces the same
output a real LLM-driven extraction over ``sample_docs/`` would
generate, but without spending API tokens. Run with:

    python -m vertexiq.seed_graph

It overwrites ``graph.json`` in the current working directory.
"""

from __future__ import annotations

try:  # pragma: no cover - import path resolution
    from .graph_builder import KnowledgeGraph
except ImportError:  # pragma: no cover
    from graph_builder import KnowledgeGraph  # type: ignore[no-redef]


# Triples grouped roughly by source document for readability.
DEMO_TRIPLES: list[dict] = [
    # --- doc1: acquisitions and partnerships ---
    {"subject": "OpenAI", "predicate": "acquired", "object": "Rockset"},
    {"subject": "OpenAI", "predicate": "founded_by", "object": "Sam Altman"},
    {"subject": "OpenAI", "predicate": "founded_by", "object": "Greg Brockman"},
    {"subject": "OpenAI", "predicate": "founded_by", "object": "Ilya Sutskever"},
    {"subject": "OpenAI", "predicate": "founded_by", "object": "Elon Musk"},
    {"subject": "Microsoft", "predicate": "invested_in", "object": "OpenAI"},
    {"subject": "Microsoft", "predicate": "partnered_with", "object": "OpenAI"},
    {"subject": "Microsoft", "predicate": "released", "object": "Phi-3"},
    {"subject": "Anthropic", "predicate": "founded_by", "object": "Dario Amodei"},
    {"subject": "Anthropic", "predicate": "founded_by", "object": "Daniela Amodei"},
    {"subject": "Amazon", "predicate": "invested_in", "object": "Anthropic"},
    {"subject": "Google", "predicate": "invested_in", "object": "Anthropic"},
    {"subject": "Anthropic", "predicate": "uses", "object": "AWS Trainium"},
    {"subject": "Google", "predicate": "partnered_with", "object": "Hugging Face"},
    {"subject": "Google DeepMind", "predicate": "released", "object": "Gemini"},
    {"subject": "Demis Hassabis", "predicate": "leads", "object": "Google DeepMind"},
    {"subject": "Meta", "predicate": "released", "object": "Llama 3"},
    {"subject": "Meta", "predicate": "partnered_with", "object": "NVIDIA"},
    {"subject": "Meta", "predicate": "partnered_with", "object": "Microsoft"},
    {"subject": "Yann LeCun", "predicate": "works_for", "object": "Meta"},
    # --- doc2: model releases ---
    {"subject": "OpenAI", "predicate": "released", "object": "GPT-4o"},
    {"subject": "GPT-4o", "predicate": "trained_on", "object": "NVIDIA H100"},
    {"subject": "GPT-4o", "predicate": "trained_in", "object": "Microsoft Azure"},
    {"subject": "Google DeepMind", "predicate": "released", "object": "Gemini 1.5 Pro"},
    {"subject": "Gemini 1.5 Pro", "predicate": "trained_on", "object": "TPUv5"},
    {"subject": "Anthropic", "predicate": "released", "object": "Claude 3 Opus"},
    {"subject": "Anthropic", "predicate": "released", "object": "Claude 3 Sonnet"},
    {"subject": "Anthropic", "predicate": "released", "object": "Claude 3 Haiku"},
    {"subject": "Anthropic", "predicate": "released", "object": "Claude 3.5 Sonnet"},
    {"subject": "Meta", "predicate": "released", "object": "Llama 3.1 405B"},
    {"subject": "Mistral AI", "predicate": "founded_by", "object": "Arthur Mensch"},
    {"subject": "Mistral AI", "predicate": "released", "object": "Mistral Large"},
    {"subject": "Mistral AI", "predicate": "released", "object": "Codestral"},
    {"subject": "Mistral AI", "predicate": "partnered_with", "object": "Microsoft"},
    {"subject": "xAI", "predicate": "founded_by", "object": "Elon Musk"},
    {"subject": "xAI", "predicate": "released", "object": "Grok-1"},
    {"subject": "xAI", "predicate": "released", "object": "Grok-2"},
    {"subject": "xAI", "predicate": "built", "object": "Colossus"},
    {"subject": "Colossus", "predicate": "uses", "object": "NVIDIA H100"},
    # --- doc3: regulations and governance ---
    {"subject": "European Union", "predicate": "finalized", "object": "EU AI Act"},
    {"subject": "EU AI Act", "predicate": "regulates", "object": "OpenAI"},
    {"subject": "EU AI Act", "predicate": "regulates", "object": "Anthropic"},
    {"subject": "EU AI Act", "predicate": "regulates", "object": "Google DeepMind"},
    {"subject": "EU AI Act", "predicate": "regulates", "object": "Meta"},
    {"subject": "European Commission", "predicate": "established", "object": "AI Office"},
    {"subject": "Joe Biden", "predicate": "signed", "object": "Executive Order 14110"},
    {"subject": "Executive Order 14110", "predicate": "directs", "object": "NIST"},
    {"subject": "Executive Order 14110", "predicate": "created", "object": "US AI Safety Institute"},
    {"subject": "US AI Safety Institute", "predicate": "under", "object": "NIST"},
    {"subject": "OpenAI", "predicate": "removed", "object": "Sam Altman"},
    {"subject": "Satya Nadella", "predicate": "backed", "object": "Sam Altman"},
    {"subject": "OpenAI", "predicate": "reinstated", "object": "Sam Altman"},
    {"subject": "Bret Taylor", "predicate": "chairs", "object": "OpenAI"},
    {"subject": "Larry Summers", "predicate": "directs", "object": "OpenAI"},
    {"subject": "OpenAI", "predicate": "dissolved", "object": "Superalignment Team"},
    {"subject": "Superalignment Team", "predicate": "led_by", "object": "Ilya Sutskever"},
    {"subject": "Superalignment Team", "predicate": "led_by", "object": "Jan Leike"},
    {"subject": "Rishi Sunak", "predicate": "hosted", "object": "Bletchley Park AI Safety Summit"},
    {"subject": "UK AI Safety Institute", "predicate": "partnered_with", "object": "US AI Safety Institute"},
    {"subject": "Gavin Newsom", "predicate": "vetoed", "object": "SB 1047"},
    {"subject": "Scott Wiener", "predicate": "authored", "object": "SB 1047"},
    {"subject": "OpenAI", "predicate": "lobbied_against", "object": "SB 1047"},
    {"subject": "Google", "predicate": "lobbied_against", "object": "SB 1047"},
    {"subject": "Meta", "predicate": "lobbied_against", "object": "SB 1047"},
]


def main(path: str = "graph.json") -> None:
    """Build and persist the demo graph at ``path``."""
    g = KnowledgeGraph(path=path)
    # Reset to a clean state so the seed is deterministic.
    g.graph.clear()
    added = g.add_triples(DEMO_TRIPLES)
    g.save()
    stats = g.get_stats()
    print(
        f"Seeded {path}: {stats['nodes']} nodes, {stats['edges']} edges "
        f"(added {added} of {len(DEMO_TRIPLES)} triples)."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
