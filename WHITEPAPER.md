# Personal Memory as a Knowledge Graph: A Node-Centric Architecture for LLM Memory Systems

---

## Abstract

Large language models (LLMs) have no persistent memory across conversations. Existing approaches to memory — flat key-value stores, vector embeddings, or full conversation logs — either lose structural relationships between facts or require expensive retrieval operations that return world knowledge the model already possesses. This paper proposes a **personal knowledge graph** architecture designed specifically as an LLM memory system. The core principle is the separation of two fundamentally different types of knowledge: *world knowledge* (facts the LLM already knows) and *personal knowledge* (facts about a specific user that no LLM could know). The system stores and infers only personal knowledge, using deterministic grammar-based parsing to extract facts from natural language and a node-centric bidirectional graph model that supports O(degree) traversal in both directions. We describe the data model, parsing strategy, inference rules, and traversal algorithms, and evaluate the system against a vector store baseline. On 25 retrieval scenarios requiring relational reasoning, the knowledge graph achieves 96% entity recall versus 63% for vector similarity — a +33 point advantage driven by the graph's ability to traverse multi-hop relationships, compute set intersections, and distinguish active from superseded facts. A self-improving rule cache reduces LLM calls by 72% compared to always-call-LLM extraction, converging toward zero cost over time.

---

## 1. Introduction

A user tells an LLM: *"My mother Sunita lives in Delhi."* A week later, in a new conversation, the user says: *"I'm thinking of visiting family."* For the LLM to make the connection — to understand that this likely means visiting Sunita in Delhi — it needs memory that spans sessions.

Current LLM memory approaches fall into three categories:

1. **Conversation logs** — store the full text of prior conversations. Expensive to retrieve. Contains irrelevant context. No structure.

2. **Vector stores** — embed facts and retrieve by semantic similarity. Effective for fuzzy recall. Loses relational structure: "Sunita lives in Delhi" and "Nikhil visits Delhi every Diwali" are stored as independent vectors with no connection between them.

3. **Flat key-value memory** — store extracted facts as strings. Fast. But unstructured: no way to traverse *from* Sunita *to* Nikhil *to* Diwali.

None of these approaches exploit the **relational structure** that makes human memory powerful. When a person hears "Sunita," their brain immediately activates everything connected: *she's Nikhil's mother, she lives in Delhi, Nikhil visits every Diwali, she's a retired teacher.* This associative, bidirectional traversal is what a knowledge graph can provide — but only if designed correctly.

Existing knowledge graph systems (RDF, Neo4j, Wikidata) are general-purpose and store everything, including world knowledge. Storing "biryani is Indian food" is redundant when the LLM already knows this. The opportunity is to build a *specialized* knowledge graph that stores only what the LLM *cannot* know: personal facts, relationships, timelines, and preferences specific to the user.

This paper describes such a system, covering:
- The philosophical boundary between world knowledge and personal knowledge
- A node-centric bidirectional data model with O(degree) traversal
- Deterministic grammar-based parsing to extract personal facts from natural language
- A personal inference engine that derives new connections from existing ones without world knowledge
- Placeholder nodes for partial information and merge operations for progressive resolution

---

## 2. Background and Related Work

### 2.1 Knowledge Graphs

A knowledge graph represents information as a set of triples: *(subject, predicate, object)*. This formalism originates in RDF (Resource Description Framework) and underlies systems such as Wikidata, DBpedia, and Google's Knowledge Graph. In these systems, both world knowledge ("Paris is the capital of France") and personal knowledge ("Alice lives in Paris") coexist. For LLM memory, mixing these creates noise: the LLM already knows Paris is a capital city — it does not need us to tell it.

### 2.2 Property Graphs

Property graphs (Neo4j, Amazon Neptune) extend the triple model by allowing nodes and edges to carry arbitrary key-value properties. This is closer to our model: an edge can carry a date, a qualifier, and a confidence score alongside its subject-predicate-object structure. Our Edge dataclass is essentially a property graph edge.

### 2.3 LLM Memory Systems

MemGPT [Packer et al., 2023] introduces a tiered memory system for LLMs with in-context and external storage but relies on flat text retrieval. MemoryBank [Zhong et al., 2023] uses vector stores with temporal decay. Neither exploits relational structure for traversal.

### 2.4 Dependency Parsing for Information Extraction

Open Information Extraction (OpenIE) [Banko et al., 2007] extracts (subject, relation, object) triples from text using patterns. Our parser goes further: it uses spaCy's full dependency parse tree to extract not just the triple but its modifiers — qualifiers (adverbs), conditions (prepositional phrases), temporals (date NER), and negation — attaching them as properties on the edge rather than losing them.

---

## 3. The Core Principle: World vs. Personal Knowledge

The single most important design decision in this system is the boundary between what the graph stores and what the LLM already knows.

### 3.1 World Knowledge (LLM already has this)

- Biryani is Indian food
- Mumbai is a city in India
- Python is a programming language
- Doctors work in hospitals
- Delhi is the capital of India

Storing or inferring any of this in the graph wastes space and inference cycles. When the LLM reads "Nikhil loves biryani" from the graph, it will *already* connect biryani to Indian cuisine. The graph need not do this work.

### 3.2 Personal Knowledge (Only the graph has this)

- Sunita is Nikhil's mother
- Nikhil moved from Bangalore to Mumbai on July 15, 2026
- Nikhil is allergic to peanuts
- Nikhil visits Delhi every Diwali
- Nikhil left Flipkart and joined Razorpay
- Nikhil leads (led) the payments team

This is information no LLM can derive. It is specific to one user's life, relationships, and history. This is what the graph exists to store.

### 3.3 The Inference Boundary

This principle extends to inference: the graph should only infer **personal connections** — links between two personal facts that the LLM could not make without both facts being present in the graph simultaneously.

**Valid personal inference:**
> Nikhil visits Delhi every Diwali (personal fact) + Sunita lives in Delhi (personal fact) + Sunita is Nikhil's mother (personal fact) → Nikhil likely visits his mother every Diwali

The LLM cannot make this connection without all three facts. The inference is valuable.

**Invalid world-knowledge inference:**
> Nikhil loves biryani → biryani is Indian food → infer Nikhil likes Indian food

The LLM makes this inference automatically from "Nikhil loves biryani." Storing it in the graph is redundant.

---

## 4. Data Model

### 4.1 Node

Every entity in the graph — person, place, organization, or concept — is a Node:

```
Node {
    name       : str            # canonical lowercase identifier
    type       : str            # PERSON | PLACE | ORG | CONCEPT
    attributes : list[str]      # permanent descriptors ("35 years old", "software engineer")
    states     : list[State]    # temporary states with date and cause
    aliases    : list[str]      # resolved placeholder names ("nikhil:mother")
    edges_out  : list[EdgeID]   # IDs of edges where this node is SUBJECT
    edges_in   : list[EdgeID]   # IDs of edges where this node is OBJECT
}
```

`edges_out` and `edges_in` are the key innovation. Every node is its own access point in both directions. A single hashmap lookup `node_store["sunita"]` immediately yields everything connected to Sunita — who she is connected to, who is connected to her — without any additional index traversal.

### 4.2 Edge

Every relationship is an Edge stored once in a flat edge store:

```
Edge {
    id         : str       # unique identifier ("e1", "e2", ...)
    subject    : str       # node name
    predicate  : str       # LIKES | LIVES_IN | MOTHER | WORKS_AT | LEADS | ...
    object     : str       # node name
    date       : str       # date this fact was stated/known
    qualifier  : str       # frequency/manner: "every morning", "a lot"
    condition  : str       # context: "for transactional workloads", "in 2027"
    intensity  : str       # comparative: "over vscode", "more than sushi"
    polarity   : bool      # True = affirmed, False = negated
    active     : bool      # False = superseded (history preserved)
    inferred   : bool      # True = derived by inference engine, not stated
    confidence : float     # 1.0 deterministic, < 1.0 probabilistic
}
```

### 4.3 Storage

Two hashmaps constitute the entire storage:

```
node_store : dict[str, Node]   # entity_name → Node
edge_store : dict[str, Edge]   # edge_id → Edge
```

Every edge is stored exactly once. Forward and backward adjacency is embedded in the nodes as lists of edge IDs — these are pointers, not copies. Total space is O(V + E).

### 4.4 Why This Replaces Six Hashmaps

The previous design used six separate hashmaps that fragmented a single logical fact:

| Old structure | New location |
|---|---|
| `entity_map[name] → type` | `node_store[name].type` |
| `subject_map[name] → [verb_keys]` | `node_store[name].edges_out` |
| `verb_map[verb_key] → [objects]` | `edge_store[edge_id]` |
| `object_map[obj] → [verb_keys]` | `node_store[name].edges_in` |
| `info_map[name] → [attrs]` | `node_store[name].attributes` |
| `category_map` | **Removed** (world knowledge) |

The old model had no way to answer "who likes pizza?" in better than O(V) because the verb_key did not encode the subject. The new model answers this in O(degree) via `pizza.edges_in`.

---

## 5. Parsing and Information Extraction

### 5.1 Philosophy

Facts arrive as natural language sentences curated by the LLM. The LLM is responsible for selecting which facts are worth storing; the parsing layer is responsible for converting them into graph operations with no ambiguity.

The pipeline is **deterministic first, probabilistic only when unavoidable**. Grammar rules handle the structural extraction. The LLM handles only judgments that grammar cannot resolve.

### 5.2 The Parsing Pipeline

**Step 1 — spaCy dependency parse**
Using `en_core_web_lg`, each sentence is parsed into a full dependency tree with POS tags, NER labels, and morphological features.

**Step 2 — Post-parse fixer (custom spaCy component)**
spaCy's statistical parser makes systematic errors on certain sentence structures. A rule-based post-parse component corrects three known patterns before downstream processing:

- *Pattern A*: Root is a noun with a VERB compound/amod child in VERB_EDGE_MAP → swap to make the verb the root. Fixes "Nikhil dislikes coffee" where spaCy may parse "coffee" as root with "dislikes" as compound.
- *Pattern B*: Root is VBN (past participle) in STATE_ADJECTIVES with an auxpass "be" child → swap to make "be" the root. Fixes "Nikhil is stressed" where "stressed" may be parsed as root.
- *Pattern C*: Root is NNS (noun plural) that lemmatizes as a known verb under forced VERB POS → re-tag as VBZ. Fixes "Nikhil lives in Mumbai" where "lives" may be tagged as noun.

**Step 3 — Atom extraction**
From the corrected parse tree, the system extracts *atoms* — minimal unit operations on the graph. Verb resolution checks two sources in order: the static `VERB_EDGE_MAP` lookup table and the **learned rules cache** (see Section 5.6):

| Atom type | Example | Source structure |
|---|---|---|
| `verb_relation` | Nikhil likes biryani | nsubj + ROOT(VERB) + dobj/pobj |
| `attribute` | Nikhil is a software engineer | nsubj + is(ROOT) + attr |
| `state` | Nikhil is stressed | nsubj + is(ROOT) + acomp in STATE_ADJECTIVES |
| `possessive` | Nikhil's mother lives in Delhi | poss dependency |
| `modal_category` | Biryani could be Indian food | modal + be |
| `unknown_verb` | Nikhil admires Elon Musk | verb not in lookup or learned rules → LLM fallback |

**Step 4 — Conjunct expansion**
Sentences with multiple objects ("Nikhil loves biryani, dosa and samosa") produce multiple atoms via BFS over `conj` dependency children. Sentences with multiple attributes ("Nikhil is a software engineer and a food lover") expand via `conj` and `appos` children of the primary `attr` token.

**Step 5 — Learned prepositional pattern scan**
After main extraction, the parser scans for prepositional patterns from the learned rules cache. This handles structures that are not verb-based — for example, "into running" → `INTERESTED_IN` — where the predicate is carried by a preposition rather than a verb. Only prepositions structurally related to the root (direct children or conjuncts) are considered. See Section 5.6 for the learning mechanism.

**Step 6 — LLM fallback for unknown patterns**
If any `unknown_verb` atoms remain after Steps 3–5, the sentence is sent to an LLM (Gemini Flash via free tier) for parsing. The LLM returns both the parsed atoms and **generalizable rules** that are cached for future use. This step is detailed in Section 5.6.

### 5.3 Possessive Handling

"Nikhil's mother lives in Delhi" is a two-triple sentence:
1. Nikhil → MOTHER → [unnamed entity]
2. [unnamed entity] → LIVES_IN → Delhi

The possessive is detected via `token.dep_ == "poss"`. The owned noun is looked up in `POSSESSIVE_PREDICATES` to determine the relationship type. If the entity has no name, a placeholder node is created: `"nikhil:mother"`.

### 5.4 Qualifier Extraction

Adverbs and prepositional phrases that modify the verb become edge qualifiers:

- Frequency: "Nikhil *sometimes* eats pizza" → qualifier="sometimes"
- Manner: "Nikhil likes biryani *a lot*" → qualifier="a lot"
- Condition: "Nikhil prefers PostgreSQL *for transactional workloads*" → condition="for transactional workloads"
- Comparative: "Nikhil prefers vim *over vscode*" → intensity="over vscode"
- Temporal: "Nikhil liked pizza *yesterday*" → date="yesterday"

DATE and TIME NER entities are extracted as the edge date, not as qualifiers.

### 5.6 Self-Improving Rule Cache

The system implements a three-layer parsing architecture that reduces LLM calls over time:

**Layer 1 — Static lookup (`VERB_EDGE_MAP`).** Hand-written verb→predicate mappings: `like→LIKES`, `eat→EATS`, etc. These are deterministic and never change at runtime.

**Layer 2 — Learned rules (`learned_rules.json`).** Rules taught by the LLM on prior encounters, persisted to disk. Two rule types:

- *Verb mappings*: `{verb_lemma: "PREDICATE"}` — extends Layer 1 at runtime. Example: after the LLM processes "Nikhil admires Elon Musk", it teaches `admire→ADMIRES`. All future uses of "admire" resolve deterministically without an LLM call.

- *Prepositional pattern rules*: `{prep: "into", predicate: "INTERESTED_IN"}` — structural patterns that are not verb-based. Example: after the LLM processes "Nikhil is into running", it teaches that `into + noun → INTERESTED_IN`. All future "into X" patterns resolve deterministically.

**Layer 3 — LLM call (Gemini Flash, free tier).** Invoked only when Layers 1 and 2 cannot resolve a sentence. The LLM is prompted to return both the parsed atoms (in the same format as the deterministic parser) and generalizable rules. Rules are cached in Layer 2 immediately, so the same structural pattern never triggers a second LLM call.

The key property is that **LLM calls decrease monotonically over time**. Every LLM invocation teaches one or more rules. Once learned, those rules handle all future instances of the same pattern without any LLM involvement. In practice, the system converges quickly: after processing a few dozen diverse sentences, most natural language patterns are covered by the learned rules cache.

The LLM prompt explicitly instructs the model to return *general* rules, not sentence-specific mappings. "do + activity → PRACTICES" applies to yoga, karate, meditation, and any future activity noun. Common auxiliary verbs (`be`, `have`, `do`, `will`, etc.) are filtered from verb mappings on the receiving end, preventing overly broad rules that would misfire on unrelated sentences.

**Graceful degradation:** If no API key is configured or the LLM is unreachable, unknown patterns are flagged in the output but the system continues operating. No crash, no data loss.

### 5.7 Entity Type Detection

Entity type is determined from spaCy NER (PERSON, GPE→PLACE, ORG), falling back to PROPN→PERSON for unrecognized proper nouns. There is no lookup table of food or cuisine names — these are world knowledge and the entity type is simply CONCEPT.

---

## 6. Inference Engine

### 6.1 Design Constraint

**The inference engine must only infer facts the LLM could not derive without the graph.**

This constraint disqualifies all world-knowledge inference. The engine reasons only over personal facts already in the graph.

### 6.2 Rule 1 — Supersession Cascade (Deterministic)

When a `WORKS_AT` edge is deactivated (person changed jobs), roles held at that organization become inactive:

```
nikhil WORKS_AT flipkart → deactivated
→ search edges_out of nikhil for LEADS/MANAGES where object.edges_in includes flipkart
→ deactivate those edges (active=False, inferred=True, confidence=1.0)
```

This is fully deterministic: if you no longer work somewhere, you no longer lead teams there.

### 6.3 Rule 2 — Temporal Co-occurrence (Probabilistic)

When a `VISITS` edge is added to a location, check if any known PERSON has `LIVES_IN` that location AND has a family/personal relationship edge to the subject:

```
nikhil VISITS delhi [new edge]
→ search: who has LIVES_IN delhi?
→ sunita LIVES_IN delhi
→ search: does nikhil have a personal relation edge to sunita?
→ nikhil MOTHER sunita ✓
→ infer: nikhil VISITS sunita (qualifier: "every Diwali", inferred=True, confidence=0.8)
```

Confidence is 0.8 because "visits Delhi" could be for other reasons.

### 6.4 Rule 3 — Placeholder Resolution (Deterministic)

When a sentence names an entity that matches a placeholder alias:

```
"Nikhil's mother is Sunita" arrives
→ parser detects: subject="nikhil:mother", attribute="sunita" (proper noun)
→ check node_store for aliases containing "nikhil:mother"
→ found: node "nikhil:mother" (placeholder)
→ merge: create/update node "sunita", copy all edges, add "nikhil:mother" to aliases
→ delete placeholder node
```

After merge, any future reference to "nikhil:mother" resolves to "sunita".

### 6.5 Rule 4 — Contradiction Detection (Deterministic)

When adding an edge with the same subject and predicate where only one object is logically possible (LIVES_IN, WORKS_AT), supersede the prior active edge:

```
nikhil LIVES_IN bangalore [active]
nikhil LIVES_IN mumbai [new]
→ deactivate: nikhil LIVES_IN bangalore (active=False)
→ store history, activate: nikhil LIVES_IN mumbai
```

---

## 7. Traversal and Query Interface

### 7.1 Complexity Analysis

All traversal operations are O(1) + O(k) where k is the degree of the queried node, because:
- `node_store[name]` is O(1) hashmap lookup
- `node.edges_out` and `node.edges_in` are pre-built lists, not computed at query time
- Each edge in those lists is O(1) to retrieve from `edge_store`

### 7.2 Core Queries

**`about(entity)`** — everything known about an entity

```
node ← node_store[entity]
return {
    type:       node.type,
    attributes: node.attributes,
    states:     node.states (active),
    outgoing:   [edge_store[id] for id in node.edges_out if active],
    incoming:   [edge_store[id] for id in node.edges_in if active],
    history:    [edge_store[id] for id in edges_out+edges_in if not active]
}
```

O(degree). This is the "thinking of Sunita" query — one access point yields everything.

**`who_does(predicate, object)`** — reverse lookup

```
node ← node_store[object]
return [edge_store[id].subject
        for id in node.edges_in
        if edge_store[id].predicate == predicate and active]
```

O(degree(object)). Was O(V) in the old model. Now O(degree).

**`compare(entity1, entity2)`** — shared connections

```
objects1 = {edge_store[id].object for id in node1.edges_out if active}
objects2 = {edge_store[id].object for id in node2.edges_out if active}
return objects1 ∩ objects2
```

O(degree1 + degree2).

**`timeline(entity)`** — history including superseded facts

```
all_edges = [edge_store[id] for id in node.edges_out + node.edges_in]
return sorted(all_edges, key=lambda e: e.date)
```

O(degree × log(degree)).

**`path(start, end, max_depth)`** — how two entities are connected

BFS over `edges_out + edges_in` with visited set. O(V + E) worst case. Finds the shortest connection between two entities regardless of direction.

### 7.3 The Bidirectionality Principle

The most important property of this model is that **any entity is a valid query entry point in both directions**.

A user thinking of "samosa" can arrive at "Nikhil loves samosa" via `samosa.edges_in`. A user thinking of "Nikhil" arrives at "loves samosa" via `nikhil.edges_out`. The traversal is symmetric. No inverse index is needed because the inverse is already embedded in the node.

This mirrors human associative memory: recalling a concept activates all its connections regardless of which "direction" they were originally formed.

---

## 8. Key Design Decisions and Tradeoffs

### 8.1 Two Hashmaps vs. Six

The six-hashmap design was logically fragmented. A single fact required coordinated writes to subject_map, verb_map, and object_map, and reverse lookup required scanning all subjects. The two-hashmap design (node_store + edge_store) makes the fact an atomic unit while preserving O(1) lookup via the node's edge lists.

### 8.2 Deterministic-First Parsing with LLM Fallback

The system uses spaCy dependency parsing with custom post-parse fixers as the primary extraction method. Grammar rules are deterministic, reproducible, and fast — no latency, no hallucination risk, no API cost. LLM extraction is used only as a fallback for patterns the grammar rules cannot handle.

Critically, the LLM fallback is **self-eliminating**: each LLM call teaches the system a generalizable rule that is cached and applied deterministically on future encounters. The LLM is not called on every write — it is called at most once per novel pattern, then never again for that pattern. This gives the system the coverage of LLM extraction with the cost profile of deterministic parsing.

### 8.3 Placeholder Nodes for Partial Information

Rather than discarding "Nikhil's mother lives in Delhi" because we don't know the mother's name, the system creates a placeholder node `nikhil:mother`. This preserves partial information and allows progressive resolution. When the name is later revealed, merge propagates all existing edges automatically. This models how knowledge is actually acquired — in fragments over time.

### 8.4 Active/Inactive Edges for Temporal History

Superseded facts are marked `active=False` but never deleted. This preserves the full timeline: "Nikhil worked at Flipkart (2024-2026), then moved to Razorpay." The LLM can use historical context when relevant, while active-only queries return current state.

### 8.5 No DAG Enforcement on Relational Edges

Relational edges (LIKES, WORKS_AT, etc.) are allowed to form cycles. "Nikhil likes Priya, Priya likes Nikhil" is semantically valid. Traversal uses a visited set to prevent infinite loops, maintaining O(V+E) complexity. Only hierarchical IS_A edges (if used) require DAG enforcement.

---

## 9. Example: Full System Trace

**Input sequence** (curated by LLM from conversation):
```
1. "Nikhil is a 35 year old software engineer"
2. "Nikhil lives in Bangalore"
3. "Nikhil works at Flipkart"
4. "Nikhil leads the payments team at Flipkart"
5. "Nikhil's mother lives in Delhi"
6. "Nikhil visits Delhi every Diwali"
7. "Nikhil moved to Mumbai"
8. "Nikhil's mother is Sunita"
```

**Graph state after sentence 5:**
```
nodes: nikhil, bangalore, flipkart, payments_team, delhi, nikhil:mother
edges:
  e1: nikhil → LIVES_IN → bangalore
  e2: nikhil → WORKS_AT → flipkart
  e3: nikhil → LEADS → payments_team
  e4: nikhil → MOTHER → nikhil:mother
  e5: nikhil:mother → LIVES_IN → delhi
```

**After sentence 6 + inference Rule 2:**
```
  e6: nikhil → VISITS → delhi (qualifier: "every Diwali")
  e7: nikhil → VISITS → nikhil:mother (inferred, conf=0.8, qualifier: "every Diwali")
```

**After sentence 7 + inference Rule 1 cascade:**
```
  e1: nikhil → LIVES_IN → bangalore [active=False]
  e8: nikhil → LIVES_IN → mumbai [active=True]
```

**After sentence 8 + placeholder merge:**
```
  node "nikhil:mother" → merged into "sunita" (aliases: ["nikhil:mother"])
  e4: nikhil → MOTHER → sunita
  e5: sunita → LIVES_IN → delhi
  e7: nikhil → VISITS → sunita (inferred)
```

**Query: `about sunita`**
```
node_store["sunita"]
  type:       PERSON
  attributes: []
  edges_out:  [e5]  → LIVES_IN → delhi
  edges_in:   [e4, e7]  ← nikhil MOTHER sunita, nikhil VISITS sunita
  aliases:    ["nikhil:mother"]

Result: Sunita is Nikhil's mother. Lives in Delhi.
        Nikhil likely visits her every Diwali.
```

---

## 10. Evaluation

We evaluate the system on three dimensions: extraction accuracy, retrieval relevance, and rule cache convergence. Full results and reproduction instructions are in `eval/BENCHMARK_RESULTS.md`.

### 10.1 Extraction Accuracy

**Setup:** 60 sentences across 22 categories with hand-annotated ground-truth triples. Each sentence is fed through the parser and graph manager; extracted triples are compared against ground truth using case-insensitive exact match on (subject, predicate, object).

**Results:**

| Metric | Score |
|---|---|
| Precision | 0.890 |
| Recall | 0.890 |
| **F1** | **0.890** |

15 of 22 categories achieve perfect F1. The remaining failures are well-characterized: negation on non-sentiment verbs (parser only flips sentiment verbs, not action verbs), "moved to" not mapped to LIVES_IN, and prepositional patterns requiring a prior LLM encounter to learn.

### 10.2 Retrieval Relevance: Knowledge Graph vs Vector Store

**Setup:** 25 retrieval scenarios, each with 2–4 setup sentences, a natural language query, and expected entities. We compare two retrieval strategies:

- **Knowledge graph:** 2-hop bidirectional traversal from all stored entities using `edges_out` and `edges_in`.
- **Vector store baseline:** spaCy `en_core_web_lg` word vectors with cosine similarity retrieval (top-5, threshold > 0.5).

**Results:**

| System | Avg Entity Recall |
|---|---|
| **Knowledge Graph** | **0.960** |
| Vector Store | 0.633 |
| **Advantage** | **+0.327** |

The graph achieves perfect recall (1.0) on 10 of 11 categories. The vector store fails hardest on queries requiring relational reasoning:

| Query type | Graph | Vector | Why vectors fail |
|---|---|---|---|
| Multi-hop (2–3 hops) | 1.00 | 0.61 | Cannot follow relationship chains |
| Comparison (set intersection) | 1.00 | 0.50 | Cannot intersect edge sets |
| Negation-aware | 1.00 | 0.00 | Cannot distinguish positive from negative edges |
| Temporal sequence | 1.00 | 0.67 | No concept of active vs superseded |

**Illustrative case:** Query "I'm thinking about visiting family this holiday season" with stored facts "Nikhil's mother Sunita lives in Delhi" and "Nikhil visits Delhi every Diwali." The graph traverses nikhil → MOTHER → sunita → LIVES_IN → delhi and retrieves both expected entities. The vector store retrieves nothing — "visiting family" has no word overlap with "Sunita lives in Delhi."

The graph's advantage is structural, not implementation-dependent. A stronger embedding model (e.g., OpenAI ada-002) would narrow the gap on surface-similarity queries but cannot close it on multi-hop traversal, set intersection, or temporal reasoning — these require relational structure that vectors do not encode.

### 10.3 Rule Cache Convergence

**Setup:** 21 sentences fed sequentially into a clean system. Sentences are structured in rounds: novel patterns (should trigger LLM), reuses of learned patterns (should not), and static-vocabulary sentences (never need LLM).

**Results:**

| Metric | Value |
|---|---|
| LLM calls | 5 of 21 (23.8%) |
| Cache hit rate | 76.2% |
| vs always-call-LLM | **72% fewer calls** |

LLM calls are front-loaded: 5 calls in the first 15 sentences to learn novel verbs (admire, explore, mentor, collect, teach). The final 6 sentences all resolve from cache with zero LLM calls. Each learned rule applies to all future sentences with the same verb — "admire" is learned once and handles "admires Gandhi," "admires Nikhil," "admires the architecture," etc.

### 10.4 Limitations

- **Dataset scale:** 60 extraction sentences and 25 retrieval scenarios demonstrate the architecture but are insufficient for statistical significance claims. A publication-grade evaluation would require 300+ annotated sentences with inter-annotator agreement.
- **Vector baseline:** Uses spaCy word vectors, not a production embedding model. A stronger baseline would narrow the retrieval gap but not eliminate the structural advantages.
- **No end-to-end conversation evaluation:** We measure extraction and retrieval in isolation, not downstream conversation quality with vs without memory.
- **Single annotator:** Ground truth is author-annotated without inter-annotator agreement scoring.

---

## 11. Future Work

### 11.1 LLM-Assisted Probabilistic Inference
The self-improving rule cache (Section 5.6) demonstrates that LLM calls can teach the system reusable rules. The same principle could extend to inference: for inferences that grammar rules cannot resolve — "if Nikhil's wife is a doctor and works night shifts, she probably works at a hospital" — a separate LLM call can evaluate candidate inferences, and the reasoning pattern can be cached as a learned inference rule. The graph passes two personal facts; the LLM judges whether to store the derived connection and teaches the system the general pattern. This is distinct from world knowledge inference: the LLM is reasoning over personal data it received from the graph, not adding facts it already knew.

### 11.2 Query-Time Inference
The current inference engine runs on write. Some inferences are more efficiently computed at query time when the full context is known: "given Nikhil is allergic to peanuts and we are discussing restaurants in Bangkok, flag dishes with peanuts." This requires integrating the graph with the LLM's current conversation context.

### 11.3 Confidence Decay
Probabilistic edges (confidence < 1.0) should decay over time unless confirmed. "Nikhil probably visits his mother every Diwali" becomes less certain if years pass without confirmation.

### 11.4 Multi-Subject Graphs
The current design assumes one user (the person whose memory is being stored). Extending to multi-user scenarios — where Nikhil, Priya, and Sunita each have their own perspectives — requires per-subject confidence scores and conflict resolution.

### 11.5 Traversal for Context Retrieval
A key open problem is determining *which* facts to retrieve and pass to the LLM for a given conversation turn. Full graph traversal is too expensive and returns too much. Topic-sensitive activation — starting traversal from nodes most relevant to the current conversation topic — is an active area of design.

---

## 12. Conclusion

We have described a knowledge graph architecture designed specifically as a personal memory system for LLMs. The central insight is that LLMs already possess vast world knowledge; what they lack is *personal* knowledge about the individual user. By restricting the graph to personal facts, using grammar-based deterministic parsing for extraction, embedding bidirectionality in every node, and limiting inference to personal connections, the system achieves:

- **O(degree) traversal** in both directions from any node
- **No redundant world knowledge storage**
- **Progressive resolution** of partial information via placeholder nodes
- **Full temporal history** with supersession
- **Deterministic-first extraction** using grammar rules, with a self-improving LLM fallback that learns generalizable rules and converges toward zero LLM calls over time

Empirical evaluation (Section 10) demonstrates the practical impact of these design decisions: the system extracts personal facts at 89% F1, retrieves relevant entities with 96% recall (vs 63% for vector similarity baselines), and reduces LLM dependency by 72% through its self-improving rule cache.

The architecture is not a general-purpose knowledge graph. It is purpose-built for one thing: giving an LLM the ability to remember the person it is talking to.

---

## References

- Banko, M., et al. (2007). Open information extraction from the web. *IJCAI*.
- Packer, C., et al. (2023). MemGPT: Towards LLMs as operating systems. *arXiv:2310.08560*.
- Zhong, W., et al. (2023). MemoryBank: Enhancing large language models with long-term memory. *arXiv:2305.10250*.
- Suchanek, F., et al. (2007). YAGO: A core of semantic knowledge. *WWW*.
- Miller, G. (1995). WordNet: A lexical database for English. *Communications of the ACM*.
- Honnibal, M., et al. (2020). spaCy: Industrial-strength natural language processing in Python.
- Ehrlinger, L., & Wöß, W. (2016). Towards a definition of knowledge graphs. *SEMANTiCS*.
