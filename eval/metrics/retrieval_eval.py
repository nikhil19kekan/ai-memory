"""
Retrieval relevance benchmark.
Tests whether the right entities are surfaced for a given query.

Compares:
  1. Knowledge graph traversal (our system)
  2. Vector store (cosine similarity baseline)

Metric: entity recall — what fraction of expected entities appear in retrieved results.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from graph_manager import GraphManager
import traversal as tv


def load_dataset(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Knowledge Graph retrieval ──────────────────────────────────

def setup_graph(sentences):
    """Feed sentences into a fresh graph, return the GraphManager."""
    gm = GraphManager()
    for s in sentences:
        gm.process(s)
    return gm


def retrieve_graph(gm, query_entities):
    """
    Given known entities from the query, retrieve all connected entities
    via graph traversal (about + 1-hop expansion).
    Returns set of entity names found.
    """
    found = set()
    for entity in query_entities:
        node = gm.graph.get_node(entity)
        if not node:
            continue
        found.add(node.name)
        # add all directly connected entities
        for edge in gm.graph.get_edges_out(entity):
            found.add(edge.object)
        for edge in gm.graph.get_edges_in(entity):
            found.add(edge.subject)
        # 2-hop: expand neighbors
        for edge in gm.graph.get_edges_out(entity):
            for e2 in gm.graph.get_edges_out(edge.object):
                found.add(e2.object)
            for e2 in gm.graph.get_edges_in(edge.object):
                found.add(e2.subject)
        for edge in gm.graph.get_edges_in(entity):
            for e2 in gm.graph.get_edges_out(edge.subject):
                found.add(e2.object)
            for e2 in gm.graph.get_edges_in(edge.subject):
                found.add(e2.subject)

    return found


def retrieve_graph_full(gm):
    """Return all entity names in the graph (for recall ceiling check)."""
    return set(gm.graph.node_store.keys())


# ── Vector store baseline ──────────────────────────────────────

def build_vector_store(sentences):
    """
    Simple vector baseline using spaCy word vectors.
    Each sentence is stored as its vector. Retrieval = cosine similarity to query.
    """
    try:
        import spacy
        import numpy as np
    except ImportError:
        return None, None

    nlp = spacy.load("en_core_web_lg")
    docs = [nlp(s) for s in sentences]
    vectors = [doc.vector for doc in docs]
    return nlp, list(zip(sentences, vectors))


def retrieve_vector(nlp, store, query, top_k=5):
    """Retrieve top-k most similar stored sentences to query via cosine similarity."""
    import numpy as np

    if not store:
        return set()

    query_vec = nlp(query).vector
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return set()

    scores = []
    for sentence, vec in store:
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0:
            scores.append(0)
        else:
            scores.append(np.dot(query_vec, vec) / (query_norm * vec_norm))

    top_indices = np.argsort(scores)[-top_k:][::-1]
    retrieved_sentences = [store[i][0] for i in top_indices if scores[i] > 0.5]

    # extract entity-like words from retrieved sentences (proper nouns, lowercased)
    found = set()
    for sent in retrieved_sentences:
        doc = nlp(sent)
        for ent in doc.ents:
            found.add(ent.text.lower())
        for tok in doc:
            if tok.pos_ == "PROPN":
                found.add(tok.text.lower())

    return found


# ── Evaluation ─────────────────────────────────────────────────

def entity_recall(retrieved, expected):
    """What fraction of expected entities appear in retrieved set."""
    if not expected:
        return 1.0
    hits = sum(1 for e in expected if e in retrieved)
    return hits / len(expected)


def run(dataset_path=None):
    if dataset_path is None:
        dataset_path = os.path.join(
            os.path.dirname(__file__), "../datasets/retrieval_scenarios.jsonl"
        )

    dataset = load_dataset(dataset_path)

    graph_scores = []
    vector_scores = []
    details = []

    for entry in dataset:
        sentences = entry["setup_sentences"]
        query = entry["query"]
        expected = [e.lower() for e in entry["expected_entities"]]

        # ── Graph retrieval ──
        gm = setup_graph(sentences)
        # use all entities in graph as query anchors (simulating topic detection)
        all_entities = list(gm.graph.node_store.keys())
        graph_retrieved = retrieve_graph(gm, all_entities)
        graph_recall = entity_recall(graph_retrieved, expected)
        graph_scores.append(graph_recall)

        # ── Vector retrieval ──
        nlp, store = build_vector_store(sentences)
        if nlp and store:
            vector_retrieved = retrieve_vector(nlp, store, query)
            vector_recall = entity_recall(vector_retrieved, expected)
        else:
            vector_retrieved = set()
            vector_recall = 0.0
        vector_scores.append(vector_recall)

        details.append({
            "id": entry["id"],
            "category": entry["category"],
            "query": query,
            "expected": expected,
            "graph_found": sorted(graph_retrieved & set(expected)),
            "graph_missed": sorted(set(expected) - graph_retrieved),
            "vector_found": sorted(vector_retrieved & set(expected)),
            "vector_missed": sorted(set(expected) - vector_retrieved),
            "graph_recall": graph_recall,
            "vector_recall": vector_recall,
            "hops": entry.get("hops", "?"),
        })

    avg_graph = sum(graph_scores) / len(graph_scores) if graph_scores else 0
    avg_vector = sum(vector_scores) / len(vector_scores) if vector_scores else 0

    print("=" * 70)
    print("RETRIEVAL BENCHMARK: Knowledge Graph vs Vector Store")
    print("=" * 70)
    print(f"\n  Graph avg recall:  {avg_graph:.3f}")
    print(f"  Vector avg recall: {avg_vector:.3f}")
    print(f"  Advantage:         {avg_graph - avg_vector:+.3f}")

    # by category
    cats = {}
    for d in details:
        cat = d["category"]
        if cat not in cats:
            cats[cat] = {"graph": [], "vector": []}
        cats[cat]["graph"].append(d["graph_recall"])
        cats[cat]["vector"].append(d["vector_recall"])

    print(f"\n  {'Category':<25s} {'Graph':>8s} {'Vector':>8s} {'Δ':>8s}")
    print(f"  {'─' * 51}")
    for cat, scores in sorted(cats.items()):
        g = sum(scores["graph"]) / len(scores["graph"])
        v = sum(scores["vector"]) / len(scores["vector"])
        delta = g - v
        marker = " ◀ graph wins" if delta > 0.1 else ""
        print(f"  {cat:<25s} {g:>8.2f} {v:>8.2f} {delta:>+8.2f}{marker}")

    # show cases where graph wins big or loses
    print(f"\n{'─' * 70}")
    print("NOTABLE CASES:")
    print(f"{'─' * 70}")
    for d in details:
        delta = d["graph_recall"] - d["vector_recall"]
        if delta > 0.3:
            print(f"\n  [#{d['id']}] GRAPH WINS ({d['hops']}-hop) — {d['query']}")
            print(f"    graph found: {d['graph_found']}  missed: {d['graph_missed']}")
            print(f"    vector found: {d['vector_found']}  missed: {d['vector_missed']}")
        elif delta < -0.1:
            print(f"\n  [#{d['id']}] VECTOR WINS — {d['query']}")
            print(f"    graph found: {d['graph_found']}  missed: {d['graph_missed']}")
            print(f"    vector found: {d['vector_found']}  missed: {d['vector_missed']}")

    return {
        "graph_avg_recall": avg_graph,
        "vector_avg_recall": avg_vector,
        "details": details,
    }


if __name__ == "__main__":
    run()
