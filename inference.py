"""
Personal inference engine — replaces correlation.py.
Derives personal connections from graph structure.
Never infers world knowledge the LLM already knows.

Rules:
  1. Supersession cascade — when WORKS_AT is superseded, deactivate LEADS/MANAGES at same org
  2. Temporal co-occurrence — A VISITS city + B LIVES_IN city + A knows B → infer A VISITS B
"""

from lookups import SUPERSESSION_CASCADE


def run(graph, new_edge, superseded_ids: list, date: str) -> list[str]:
    """
    Run all inference rules after a new edge is added.

    Args:
        graph:          KnowledgeGraph instance
        new_edge:       the Edge object that was just created
        superseded_ids: list of edge IDs deactivated by supersession
        date:           current date string for any inferred edges

    Returns:
        list of human-readable strings describing what was inferred.
    """
    changes = []

    # Rule 1: supersession cascade
    for eid in superseded_ids:
        old_edge = graph.edge_store.get(eid)
        if old_edge:
            changes.extend(_cascade(graph, old_edge))

    # Rule 2: temporal co-occurrence (VISITS + LIVES_IN → infer VISITS person)
    if new_edge.predicate == "VISITS" and new_edge.active and new_edge.polarity:
        changes.extend(_visit_cooccurrence(graph, new_edge, date))

    return changes


def _cascade(graph, deactivated_edge) -> list[str]:
    """
    Rule 1: When a key edge (e.g. WORKS_AT) is superseded, deactivate related
    role edges at the same object (e.g. LEADS, MANAGES at the same org).
    """
    cascade_preds = SUPERSESSION_CASCADE.get(deactivated_edge.predicate, [])
    if not cascade_preds:
        return []

    changes = []
    subject_node = graph.node_store.get(deactivated_edge.subject)
    if not subject_node:
        return changes

    for eid in subject_node.edges_out:
        edge = graph.edge_store.get(eid)
        if (edge
                and edge.active
                and edge.predicate in cascade_preds
                and edge.object == deactivated_edge.object):
            edge.active   = False
            edge.inferred = True
            changes.append(
                f"cascade: {edge.subject} --[{edge.predicate}]--> {edge.object} "
                f"deactivated (left {deactivated_edge.object})"
            )

    return changes


def _visit_cooccurrence(graph, visit_edge, date: str) -> list[str]:
    """
    Rule 2: A VISITS city + B LIVES_IN city + A and B share a personal connection
    → infer A VISITS B with confidence 0.8.

    This is personal inference: the LLM cannot know Nikhil visited his mother
    just because it knows Nikhil was in Delhi — only the graph knows both live there.
    """
    location      = visit_edge.object
    visitor       = visit_edge.subject
    changes       = []
    location_node = graph.node_store.get(location)
    if not location_node:
        return changes

    for eid in location_node.edges_in:
        resident_edge = graph.edge_store.get(eid)
        if not (resident_edge
                and resident_edge.active
                and resident_edge.predicate == "LIVES_IN"
                and resident_edge.subject != visitor):
            continue

        resident      = resident_edge.subject
        resident_node = graph.node_store.get(resident)
        if not resident_node or resident_node.type != "PERSON":
            continue

        # Only infer when there is already a known personal relationship
        # between visitor and resident — avoids spurious inferences with strangers.
        if not _are_personally_connected(graph, visitor, resident):
            continue

        # Check we haven't already inferred this edge
        existing = [
            graph.edge_store[e] for e in graph.node_store[visitor].edges_out
            if e in graph.edge_store
        ]
        already = any(
            e.predicate == "VISITS" and e.object == resident and e.inferred
            for e in existing
        )
        if already:
            continue

        graph.add_edge(
            visitor, "VISITS", resident, date,
            inferred=True, confidence=0.8,
        )
        changes.append(
            f"inferred: {visitor} --[VISITS]--> {resident} "
            f"(both connected to {location}, confidence=0.8)"
        )

    return changes


def _are_personally_connected(graph, entity1: str, entity2: str) -> bool:
    """True if entity1 and entity2 share any direct active edge in either direction."""
    node1 = graph.node_store.get(entity1)
    if not node1:
        return False
    for eid in node1.edges_out:
        e = graph.edge_store.get(eid)
        if e and e.active and e.object == entity2:
            return True
    for eid in node1.edges_in:
        e = graph.edge_store.get(eid)
        if e and e.active and e.subject == entity2:
            return True
    return False
