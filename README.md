# Knowledge Graph Memory for LLMs

LLMs have no memory. Start a new conversation and everything about you is gone — your preferences, your relationships, your history. Current solutions store memories as text chunks in vector databases and retrieve them by semantic similarity. This works for simple recall but fails the moment a question requires connecting two facts together.

This project is a **structured knowledge graph** that serves as persistent memory for LLMs. It parses natural language into a bidirectional graph of entities and relationships, enabling relational reasoning that vector stores cannot do.

## The Problem With Current Memory Systems

Mem0, Zep, MemGPT, and LangChain Memory all follow the same pattern: extract facts using an LLM, store them as text in a vector database, retrieve by semantic similarity. This works for simple recall ("what's my favorite food?") but breaks down the moment a question requires connecting two facts together.

Consider these stored memories:

```
"Nikhil's mother Sunita lives in Delhi"
"Nikhil visits Delhi every Diwali"
```

The user says: *"I'm thinking about visiting family this holiday season."*

**Mem0 / Zep / vector-backed systems** embed each memory as a vector and retrieve by cosine similarity to the query. "Visiting family" shares no words with "Sunita lives in Delhi." The retrieval returns nothing — it cannot follow the chain: family → mother is family → mother is Sunita → Sunita lives in Delhi → user visits Delhi every Diwali.

**This system** traverses the graph: `user → MOTHER → sunita → LIVES_IN → delhi`, `user → VISITS → delhi (every Diwali)`. Two hops. Both facts connected. Answer surfaced.

This isn't a contrived edge case. Any question involving connected facts — "Who do I know in London?", "What changed in my career?", "Does anyone in my family work in tech?" — requires relational traversal that similarity search cannot provide.

## Benchmarked

Mem0, Zep, and MemGPT all use vector similarity as their core retrieval mechanism. We evaluated graph traversal against vector similarity retrieval (spaCy `en_core_web_lg` word vectors) on 25 scenarios requiring relational reasoning — the category where these systems are architecturally weakest:

| System | Entity Recall |
|---|---|
| **Knowledge Graph** | **0.96** |
| Vector Store | 0.63 |

**+33 points.** The graph achieves perfect recall on every relational category:

| Query Type | Graph | Vector | Why Vectors Fail |
|---|---|---|---|
| Multi-hop (2-3 relationship hops) | 1.00 | 0.61 | Cannot follow relationship chains |
| Comparison (shared connections) | 1.00 | 0.50 | Cannot intersect edge sets |
| Negation ("doesn't like X") | 1.00 | 0.00 | Cannot distinguish positive from negative |
| Temporal ("previous job?") | 1.00 | 0.67 | No concept of current vs superseded |
| Career history (all jobs, ordered) | 1.00 | 0.67 | Oldest facts have lowest similarity |

Full benchmark methodology and results: [`eval/BENCHMARK_RESULTS.md`](eval/BENCHMARK_RESULTS.md)

## What It Does

Feed it natural language sentences. It extracts structured facts, builds a graph, runs inference, and answers traversal queries.

**Input:**
```
"Nikhil is a software engineer"
"Nikhil works at Google"
"Nikhil's mother Sunita lives in Delhi"
"Nikhil visits Delhi every Diwali"
"Nikhil likes biryani and dosa"
"Nikhil hated coffee but now loves it"
"Nikhil left Google and joined Razorpay"
```

**What the graph captures that a vector store cannot:**

- `nikhil → MOTHER → sunita → LIVES_IN → delhi` — traversable relationship chain
- `nikhil → VISITS → delhi (every Diwali)` + `sunita → LIVES_IN → delhi` → **inferred:** `nikhil → VISITS → sunita` (confidence 0.8)
- `nikhil → WORKS_AT → google` marked **superseded**, `nikhil → WORKS_AT → razorpay` is current — full career timeline preserved
- `nikhil → HATES → coffee` superseded by `nikhil → LOVES → coffee` — sentiment history tracked
- `nikhil → LIKES → biryani`, `nikhil → LIKES → dosa` — queryable in both directions: "what does nikhil like?" and "who likes biryani?" both O(degree)

## How It's Different

### vs Mem0

Mem0 extracts memories as natural language strings using an LLM on every write, stores them in a vector database, and retrieves by semantic similarity. It has deduplication and conflict detection, but memories remain flat text with no relational structure.

- Mem0 stores: `"Nikhil's mother is Sunita"` and `"Sunita lives in Delhi"` as two independent text memories
- This system stores: `nikhil → MOTHER → sunita → LIVES_IN → delhi` as a traversable chain
- Mem0 calls an LLM on every write. This system parses deterministically and calls an LLM only for novel patterns, learning a reusable rule each time (72% fewer LLM calls, converging to zero)
- Mem0 cannot answer "who does Nikhil know in Delhi?" without retrieving every memory and hoping the LLM connects the dots. This system traverses the graph in O(degree)

### vs Zep

Zep stores conversation history and extracted facts, with embedding-based retrieval. It added entity extraction, but entities are tags on memories — not nodes in a traversable graph with typed, directed, dated edges.

- Zep has no concept of supersession. If a user changes jobs, both the old and new job exist as equally valid memories. This system marks the old edge `active=False` and preserves it as history — queryable via `timeline` but excluded from current-state queries
- Zep cannot compute set intersections ("what do Nikhil and Priya both like?") or find paths between entities. These are native graph operations here

### vs MemGPT

MemGPT treats memory as an operating system problem — the LLM manages its own tiered memory (core + archival) via function calls. Architecturally interesting, but the storage is still flat text. The LLM decides what to store and retrieve, meaning every memory operation has LLM latency and cost.

- MemGPT's retrieval quality depends entirely on the LLM's ability to formulate good search queries at runtime. This system's retrieval is structural — graph traversal finds connections that no search query would surface
- MemGPT has no inference. This system derives new facts from graph structure: "Nikhil visits Delhi" + "Sunita lives in Delhi" + "Sunita is Nikhil's mother" → inferred: "Nikhil visits Sunita" (confidence 0.8)

### vs general knowledge graphs (Neo4j, Wikidata)

General-purpose knowledge graphs store everything, including world knowledge. This system stores only what the LLM doesn't already know. Storing "biryani is Indian food" is redundant — the LLM already knows this. The graph stores "Nikhil loves biryani" — information no LLM can derive without being told.

### The extraction cost problem

Every system above calls an LLM on every write to extract facts. This is expensive, slow, and introduces hallucination risk. This system uses deterministic grammar parsing (spaCy) as the primary extraction method — zero cost, zero latency, zero hallucination. An LLM is called only for patterns the grammar rules don't cover, and each call **teaches the system a reusable rule** cached to disk. Benchmarked: **72% fewer LLM calls** than always-call-LLM, converging toward zero over time.

## Architecture

Two hashmaps. That's the entire storage.

```python
node_store: dict[str, Node]   # "sunita" → Node(edges_out=[e5], edges_in=[e4, e7], ...)
edge_store: dict[str, Edge]   # "e4"     → Edge(subject="nikhil", predicate="MOTHER", object="sunita", ...)
```

Every node carries bidirectional edge lists (`edges_out`, `edges_in`). Any entity is a valid entry point for traversal in both directions. Forward lookup ("what does nikhil like?") and reverse lookup ("who likes biryani?") are both O(degree) — no full-graph scans.

Edges carry rich metadata: date, qualifier ("every morning"), polarity (affirmed/negated), active/superseded status, inferred flag, and confidence score. Superseded facts are preserved as history, not deleted.

### Parsing pipeline

```
Natural language
  → spaCy dependency parse (en_core_web_lg)
  → Post-parse fixer (corrects known spaCy mis-parses)
  → Atom extraction (subject, predicate, object, qualifier, negation, date)
  → Learned rules check (cached verb mappings + prepositional patterns)
  → LLM fallback (Gemini Flash, free tier — only for novel patterns)
      → Learns generalizable rule → cached → never calls again for same pattern
```

Extraction F1: **0.89** on 60 sentences across 22 categories (attributes, possessives, conjuncts, negation, relative clauses, temporal, supersession, etc.)

### Self-improving rule cache

The system starts with ~50 hand-written verb→predicate mappings. When it encounters an unknown pattern:

1. Calls Gemini Flash (free tier) to parse the sentence
2. Gemini returns both the parsed facts AND a generalizable rule
3. Rule is cached to disk (`learned_rules.json`)
4. All future sentences matching that pattern resolve deterministically — no LLM call

Example: first time seeing "Nikhil admires Elon Musk" → LLM teaches `admire → ADMIRES`. Every future use of "admire" (admires Gandhi, admires the architecture, etc.) resolves from cache. Benchmarked at **72% cost reduction** vs calling the LLM on every sentence.

## Quick Start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
python main.py
```

### Optional: enable LLM fallback

The system works without an API key — unknown verbs are flagged but the graph operates normally. To enable the self-improving fallback:

```bash
cp config.example.json config.json
# Add your free Gemini API key from https://aistudio.google.com/apikey
```

### CLI commands

```
<any sentence>            Parse and add to graph
about <entity>            Everything known about an entity
what does <entity> <pred> e.g. "what does nikhil like"
who <pred> <entity>       e.g. "who likes pizza"
timeline <entity>         Full history including superseded facts
compare <e1> <e2>         Shared connections between two entities
path <e1> <e2>            Shortest connection path
dump                      Print all nodes and edges
show                      Visualize the graph (matplotlib)
```

## Run Benchmarks

```bash
python eval/run_benchmarks.py              # all three benchmarks
python eval/run_benchmarks.py extraction   # parser accuracy (P/R/F1)
python eval/run_benchmarks.py retrieval    # graph vs vector store
python eval/run_benchmarks.py convergence  # rule cache learning curve (needs API key)
```

## Run Tests

```bash
python test_parser.py    # 37 parser tests
```

## Project Structure

```
├── graph.py              Core data model — Node, Edge, KnowledgeGraph
├── parser.py             spaCy deterministic parser + post-parse fixers
├── graph_manager.py      Orchestrator: parse → update → infer → LLM fallback
├── inference.py          Personal inference (supersession cascade, visit co-occurrence)
├── traversal.py          Graph queries (about, who_does, path, timeline, compare)
├── lookups.py            Static lookup tables (verb map, possessives, states)
├── learned_rules.py      Self-improving rule cache
├── llm_client.py         Gemini Flash integration
├── visualizer.py         Live graph visualization (networkx + matplotlib)
├── main.py               Interactive CLI
├── test_parser.py        Parser tests
├── WHITEPAPER.md         Full technical paper (12 sections, 3 benchmarks)
└── eval/
    ├── run_benchmarks.py
    ├── BENCHMARK_RESULTS.md
    └── datasets/          Annotated test data (60 sentences, 25 scenarios)
```

## Technical Paper

[WHITEPAPER.md](WHITEPAPER.md) covers the full design: data model, parsing strategy, inference rules, traversal algorithms, self-improving rule cache, and evaluation with benchmark results.
