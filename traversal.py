"""
Traversal queries on the KnowledgeGraph.
All operations are O(degree) — bidirectional edge lists enable efficient reverse lookup.
No O(V) scans. Any node is a valid entry point in both directions.
"""

from collections import deque
from graph import KnowledgeGraph


def about(graph: KnowledgeGraph, entity: str) -> dict:
    """
    Everything known about an entity:
    - outgoing edges (what the entity does / relates to)
    - incoming edges (what relates to the entity)
    - permanent attributes
    - active temporary states
    """
    node = graph.get_node(entity)
    if not node:
        return {"error": f"Entity '{entity}' not found"}

    out_edges = graph.get_edges_out(entity)
    in_edges  = graph.get_edges_in(entity)

    return {
        "entity":     node.name,
        "type":       node.type,
        "attributes": list(node.attributes),
        "states":     [
            {"state": s.state, "date": s.date, "cause": s.cause}
            for s in node.states if s.active
        ],
        "out_edges":  [_fmt_edge(e) for e in out_edges],
        "in_edges":   [_fmt_edge(e) for e in in_edges],
        "aliases":    list(node.aliases),
    }


def what_does(graph: KnowledgeGraph, subject: str, predicate: str) -> list[str]:
    """
    All objects of a given predicate from subject.
    e.g. what_does(graph, 'nikhil', 'LIKES') → ['pizza', 'biryani', ...]
    O(degree) — uses node.edges_out.
    """
    edges = graph.get_edges_out(subject, predicate=predicate.upper())
    return [e.object for e in edges]


def who_does(graph: KnowledgeGraph, predicate: str, obj: str) -> list[str]:
    """
    All subjects performing predicate on obj.
    e.g. who_does(graph, 'LIKES', 'pizza') → ['nikhil', 'priya', ...]
    O(degree) — uses node.edges_in (not O(V) scan).
    """
    edges = graph.get_edges_in(obj, predicate=predicate.upper())
    return [e.subject for e in edges]


def timeline(graph: KnowledgeGraph, entity: str) -> list:
    """
    Full history for an entity — all edges (including superseded), sorted newest first.
    Useful for understanding how facts about an entity changed over time.
    """
    node = graph.get_node(entity)
    if not node:
        return []

    # Collect all edges from both directions (including inactive)
    edge_ids = set(node.edges_out) | set(node.edges_in)
    edges = [graph.edge_store[eid] for eid in edge_ids if eid in graph.edge_store]

    return sorted(edges, key=lambda e: (e.date, e.id), reverse=True)


def compare(graph: KnowledgeGraph, entity1: str, entity2: str) -> dict:
    """
    Find shared connections between two entities.
    Returns objects both connect TO, and subjects that connect to BOTH.
    O(degree1 + degree2).
    """
    e1_objects  = {e.object   for e in graph.get_edges_out(entity1)}
    e2_objects  = {e.object   for e in graph.get_edges_out(entity2)}
    e1_subjects = {e.subject  for e in graph.get_edges_in(entity1)}
    e2_subjects = {e.subject  for e in graph.get_edges_in(entity2)}

    return {
        "both_connect_to": sorted(e1_objects  & e2_objects),   # shared objects
        "connected_by":    sorted(e1_subjects & e2_subjects),  # shared inbound subjects
    }


def path(graph: KnowledgeGraph, start: str, end: str, max_depth: int = 4) -> list:
    """
    BFS to find shortest connection path between two entities.
    Traverses both edge directions (undirected search over directed graph).
    Returns list of (subject, predicate, object) triples, or [] if no path.
    O(V + E) worst case, typically O(degree * depth) in a sparse personal graph.
    """
    start = start.lower().strip()
    end   = end.lower().strip()

    if not graph.get_node(start) or not graph.get_node(end):
        return []
    if start == end:
        return []

    # BFS: queue holds (current_node, path_so_far_as_list_of_triples)
    queue   = deque([(start, [])])
    visited = {start}

    while queue:
        current, current_path = queue.popleft()

        if len(current_path) >= max_depth:
            continue

        for edge in graph.get_edges_out(current):
            neighbor = edge.object
            step     = (edge.subject, edge.predicate, edge.object)
            new_path = current_path + [step]
            if neighbor == end:
                return new_path
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, new_path))

        for edge in graph.get_edges_in(current):
            neighbor = edge.subject
            step     = (edge.subject, edge.predicate, edge.object)
            new_path = current_path + [step]
            if neighbor == end:
                return new_path
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, new_path))

    return []


# ── formatting helpers ────────────────────────────────────────────

def _fmt_edge(edge) -> str:
    """Human-readable single-line edge description."""
    label = edge.predicate
    if edge.qualifier:
        label += f", {edge.qualifier}"
    if edge.condition:
        label += f", if: {edge.condition}"
    if not edge.polarity:
        label = f"NOT {label}"
    flags = []
    if not edge.active:
        flags.append("SUPERSEDED")
    if edge.inferred:
        flags.append(f"inferred({edge.confidence:.1f})")
    suffix = f"  [{', '.join(flags)}]" if flags else ""
    return f"{edge.subject} --[{label} | {edge.date}]--> {edge.object}{suffix}"


def fmt_path(path_triples: list) -> str:
    """Format a path result as a readable chain."""
    if not path_triples:
        return "(no path found)"
    parts = [path_triples[0][0]]
    for subj, pred, obj in path_triples:
        parts.append(f"--[{pred}]-->")
        parts.append(obj)
    return " ".join(parts)
