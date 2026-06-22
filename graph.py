"""
Core data model: Node and Edge as first-class objects.
Two hashmaps (node_store + edge_store) replace the old six-hashmap design.
Every node has bidirectional edge lists (edges_out, edges_in) for O(degree) traversal.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class State:
    """Temporary state of a node (stressed, sick, happy — with date and optional cause)."""
    state:  str
    date:   str
    cause:  str  = ""
    active: bool = True


@dataclass
class Edge:
    """
    A single directed relationship between two nodes.
    Stored exactly once in edge_store.
    Referenced by ID in node.edges_out (subject side) and node.edges_in (object side).
    """
    id:         str
    subject:    str           # node name (lowercase)
    predicate:  str           # LIKES, LIVES_IN, MOTHER, WORKS_AT, IS_A, …
    object:     str           # node name (lowercase)
    date:       str
    qualifier:  str   = ""    # "every morning", "a lot"
    condition:  str   = ""    # "for transactional workloads", "in 2027"
    intensity:  str   = ""    # "over vscode", "more than sushi"
    polarity:   bool  = True  # True = affirmed, False = negated
    active:     bool  = True  # False = superseded (history preserved)
    inferred:   bool  = False # True = derived by inference engine, not stated
    confidence: float = 1.0   # 1.0 deterministic, < 1.0 probabilistic


@dataclass
class Node:
    """
    An entity in the knowledge graph.  First-class object with bidirectional edge lists.
    Any node is a valid traversal entry point — O(1) lookup + O(degree) expansion
    in both forward (edges_out) and backward (edges_in) directions.
    """
    name:       str
    type:       str                              # PERSON | PLACE | ORG | CONCEPT
    attributes: list = field(default_factory=list)  # permanent descriptors
    states:     list = field(default_factory=list)  # list[State]
    aliases:    list = field(default_factory=list)  # resolved placeholder names
    edges_out:  list = field(default_factory=list)  # edge IDs where self is SUBJECT
    edges_in:   list = field(default_factory=list)  # edge IDs where self is OBJECT


# Predicates where only one active edge is allowed per subject at a time.
# Adding a new LIVES_IN supersedes the old one regardless of object.
SINGULAR_PREDICATES = {"LIVES_IN", "WORKS_AT"}

# Predicates that conflict for the SAME object (sentiment supersession).
EXCLUSIVE_PREDICATES = {
    "LIKES":    {"HATES", "DISLIKES", "AVOIDS"},
    "LOVES":    {"HATES", "DISLIKES", "AVOIDS"},
    "HATES":    {"LIKES", "LOVES"},
    "DISLIKES": {"LIKES", "LOVES"},
}


class KnowledgeGraph:
    def __init__(self):
        self.node_store: dict[str, Node] = {}
        self.edge_store: dict[str, Edge] = {}
        self._edge_counter = 0

    # ── node operations ──────────────────────────────────────────

    def add_node(self, name: str, node_type: str) -> tuple:
        """Create node if new. Returns (node, is_new)."""
        key = name.lower().strip()
        if key in self.node_store:
            return self.node_store[key], False
        node = Node(name=key, type=node_type)
        self.node_store[key] = node
        return node, True

    def get_node(self, name: str) -> Optional[Node]:
        key = name.lower().strip()
        # direct lookup
        if key in self.node_store:
            return self.node_store[key]
        # alias lookup (placeholder resolution)
        for node in self.node_store.values():
            if key in node.aliases:
                return node
        return None

    def add_attribute(self, name: str, attribute: str):
        """Append attribute string to node (no duplicates)."""
        node = self.get_node(name)
        if node and attribute.lower() not in [a.lower() for a in node.attributes]:
            node.attributes.append(attribute)

    def add_state(self, name: str, state_name: str, date: str, cause: str = ""):
        """Add a temporary state. Deactivates any prior active state with same name."""
        node = self.get_node(name)
        if not node:
            return
        for s in node.states:
            if s.state == state_name and s.active:
                s.active = False
        node.states.append(State(state=state_name, date=date, cause=cause))

    # ── edge operations ──────────────────────────────────────────

    def add_edge(self, subject: str, predicate: str, obj: str,
                 date: str, **props) -> tuple:
        """
        Create a new edge and attach it to both endpoint nodes.
        Returns (new_edge, list_of_superseded_edge_ids).

        Supersession rules:
          - SINGULAR_PREDICATES: deactivate ALL active edges with same subj+pred
          - EXCLUSIVE_PREDICATES: deactivate conflicting pred for SAME object
        """
        subject = subject.lower().strip()
        obj     = obj.lower().strip()

        # ensure nodes exist
        if subject not in self.node_store:
            self.add_node(subject, "CONCEPT")
        if obj not in self.node_store:
            self.add_node(obj, "CONCEPT")

        superseded_ids = self._find_superseded(subject, predicate, obj)
        for eid in superseded_ids:
            self.edge_store[eid].active = False

        self._edge_counter += 1
        edge = Edge(
            id         = f"e{self._edge_counter}",
            subject    = subject,
            predicate  = predicate,
            object     = obj,
            date       = date,
            qualifier  = props.get("qualifier",  ""),
            condition  = props.get("condition",  ""),
            intensity  = props.get("intensity",  ""),
            polarity   = props.get("polarity",   True),
            active     = props.get("active",     True),
            inferred   = props.get("inferred",   False),
            confidence = props.get("confidence", 1.0),
        )
        self.edge_store[edge.id] = edge
        self.node_store[subject].edges_out.append(edge.id)
        self.node_store[obj].edges_in.append(edge.id)
        return edge, superseded_ids

    def merge_nodes(self, placeholder_name: str, real_name: str, real_type: str):
        """
        Resolve a placeholder node into a named entity.
        All edges that pointed to/from placeholder are re-pointed to real_name.
        The placeholder name is stored as an alias for future lookups.
        """
        placeholder_key = placeholder_name.lower().strip()
        real_key        = real_name.lower().strip()

        placeholder = self.node_store.get(placeholder_key)
        if not placeholder:
            return  # nothing to merge

        # ensure real node exists
        if real_key not in self.node_store:
            self.add_node(real_name, real_type)
        real = self.node_store[real_key]

        if placeholder_key not in real.aliases:
            real.aliases.append(placeholder_key)

        for eid in placeholder.edges_out:
            if eid in self.edge_store:
                self.edge_store[eid].subject = real_key
                if eid not in real.edges_out:
                    real.edges_out.append(eid)

        for eid in placeholder.edges_in:
            if eid in self.edge_store:
                self.edge_store[eid].object = real_key
                if eid not in real.edges_in:
                    real.edges_in.append(eid)

        for attr in placeholder.attributes:
            self.add_attribute(real_key, attr)

        del self.node_store[placeholder_key]

    # ── query helpers ─────────────────────────────────────────────

    def get_edges_out(self, name: str, predicate: str = None,
                      active_only: bool = True) -> list:
        """Outgoing edges from node, optionally filtered. O(degree)."""
        node = self.get_node(name)
        if not node:
            return []
        edges = [self.edge_store[e] for e in node.edges_out if e in self.edge_store]
        if active_only:
            edges = [e for e in edges if e.active]
        if predicate:
            edges = [e for e in edges if e.predicate == predicate]
        return edges

    def get_edges_in(self, name: str, predicate: str = None,
                     active_only: bool = True) -> list:
        """Incoming edges to node, optionally filtered. O(degree)."""
        node = self.get_node(name)
        if not node:
            return []
        edges = [self.edge_store[e] for e in node.edges_in if e in self.edge_store]
        if active_only:
            edges = [e for e in edges if e.active]
        if predicate:
            edges = [e for e in edges if e.predicate == predicate]
        return edges

    # ── supersession ──────────────────────────────────────────────

    def _find_superseded(self, subject: str, predicate: str, obj: str) -> list:
        node = self.node_store.get(subject)
        if not node:
            return []
        superseded = []
        exclusive_with = EXCLUSIVE_PREDICATES.get(predicate, set())
        for eid in node.edges_out:
            edge = self.edge_store.get(eid)
            if not edge or not edge.active:
                continue
            if predicate in SINGULAR_PREDICATES and edge.predicate == predicate:
                superseded.append(eid)          # singular: supersede old regardless of object
            elif edge.predicate in exclusive_with and edge.object == obj:
                superseded.append(eid)          # exclusive: supersede conflicting for same obj
            elif edge.predicate == predicate and edge.object == obj:
                superseded.append(eid)          # exact duplicate
        return superseded

    # ── snapshot for visualizer ───────────────────────────────────

    def snapshot(self) -> dict:
        nodes = {
            name: {
                "type":       n.type,
                "attributes": n.attributes,
                "states":     [{"state": s.state, "date": s.date,
                                "cause": s.cause, "active": s.active}
                               for s in n.states],
                "aliases":    n.aliases,
            }
            for name, n in self.node_store.items()
        }
        edges = {
            eid: {
                "subject":    e.subject,
                "predicate":  e.predicate,
                "object":     e.object,
                "date":       e.date,
                "qualifier":  e.qualifier,
                "condition":  e.condition,
                "intensity":  e.intensity,
                "polarity":   e.polarity,
                "active":     e.active,
                "inferred":   e.inferred,
                "confidence": e.confidence,
            }
            for eid, e in self.edge_store.items()
        }
        return {"nodes": nodes, "edges": edges}
