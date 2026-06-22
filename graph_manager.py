"""
Orchestrates parsing → graph updates → inference.
Single entry point: process(text) → changes dict.
"""

from datetime import datetime
from graph import KnowledgeGraph
import parser as p
import inference
import llm_client
from learned_rules import get_rules as _get_learned_rules
from lookups import POSSESSIVE_PREDICATES

_PRONOUNS = {"he", "she", "him", "her", "his", "hers", "they", "them", "their", "theirs"}


class GraphManager:
    def __init__(self):
        self.graph         = KnowledgeGraph()
        self._last_subject = None   # most recently seen PERSON subject (for pronoun resolution)

    def process(self, text, date_override=None):
        """
        Parse text, update graph, run inference.
        Returns a changes dict describing what was added/inferred.
        """
        text = self._resolve_pronouns(text)
        date = date_override or datetime.now().strftime("%Y-%m-%d")

        changes = {
            "new_entities":  [],
            "new_edges":     [],
            "superseded":    [],
            "inferred":      [],
            "states":        [],
            "attributes":    [],
            "unknown_verbs": [],
        }

        atoms = p.parse(text, date_override)

        for atom in atoms:
            t = atom["type"]
            if t == "verb_relation":
                self._handle_verb_relation(atom, changes)
            elif t == "attribute":
                self._handle_attribute(atom, changes)
            elif t == "state":
                self._handle_state(atom, changes)
            elif t == "modal_category":
                self._handle_modal_category(atom, changes)
            elif t == "possessive":
                self._handle_possessive(atom, changes)
            elif t == "unknown_verb":
                changes["unknown_verbs"].append(atom["verb"])

        # LLM fallback for unknown verbs/patterns
        if changes["unknown_verbs"]:
            self._resolve_with_llm(text, changes, date)

        return changes

    # ── LLM fallback ────────────────────────────────────────────────

    def _resolve_with_llm(self, text, changes, date):
        """Call LLM for sentences with unknown verbs. Learn rules from response."""
        llm_atoms, learned = llm_client.parse_with_llm(
            text, subject=self._last_subject, date=date,
        )
        if not llm_atoms:
            return  # LLM unavailable or failed — unknown_verbs stay in changes

        # Learn rules for future use (filter out auxiliaries the LLM shouldn't map)
        _SKIP_VERBS = {"be", "have", "do", "will", "shall", "can", "may", "must",
                        "would", "could", "should", "get", "go"}
        rules = _get_learned_rules()
        for verb, pred in learned.get("verb_map", {}).items():
            if verb.lower() in _SKIP_VERBS:
                continue
            rules.add_verb_mapping(verb, pred)
            changes["inferred"].append(f"learned rule: {verb} → {pred}")
        for rule in learned.get("pattern_rules", []):
            rules.add_pattern_rule(rule)
            desc = rule.get("description", rule.get("type", "pattern"))
            changes["inferred"].append(f"learned rule: {desc}")

        # Process LLM-returned atoms through normal handlers
        for atom in llm_atoms:
            self._handle_verb_relation(atom, changes)

        changes["unknown_verbs"] = []  # LLM handled them

    # ── pronoun resolution ────────────────────────────────────────

    def _resolve_pronouns(self, text):
        """Replace a leading pronoun with the last known PERSON subject."""
        if not self._last_subject:
            return text
        words = text.strip().split()
        if words and words[0].lower() in _PRONOUNS:
            words[0] = self._last_subject.capitalize()
            return " ".join(words)
        return text

    # ── node registration ─────────────────────────────────────────

    def _ensure_node(self, name, node_type, changes):
        """Add node if new; register PERSON with spaCy EntityRuler."""
        node, is_new = self.graph.add_node(name, node_type)
        if is_new:
            changes["new_entities"].append(f"{name} ({node_type})")
            if node_type == "PERSON":
                p.register_person(name)
        return node, is_new

    # ── atom handlers ─────────────────────────────────────────────

    def _handle_verb_relation(self, atom, changes):
        subj      = atom["subject"]
        subj_type = atom["subject_type"]
        edge_type = atom["edge_type"]
        obj       = atom["object"]
        obj_type  = atom["object_type"]
        qualifier = atom.get("qualifier", "")
        condition = atom.get("condition", "")
        negated   = atom.get("negated", False)
        date      = atom["date"]

        self._ensure_node(subj, subj_type, changes)
        self._ensure_node(obj,  obj_type,  changes)

        if subj_type == "PERSON":
            self._last_subject = subj.lower()

        edge, superseded_ids = self.graph.add_edge(
            subj, edge_type, obj, date,
            qualifier  = qualifier,
            condition  = condition,
            polarity   = not negated,
        )

        label = f"{subj} --[{edge_type}"
        if qualifier:
            label += f", {qualifier}"
        label += f" | {date}]--> {obj}"
        changes["new_edges"].append(label)

        for eid in superseded_ids:
            e = self.graph.edge_store[eid]
            changes["superseded"].append(f"{e.subject} --[{e.predicate}]--> {e.object}")

        inferred = inference.run(self.graph, edge, superseded_ids, date)
        changes["inferred"].extend(inferred)

    def _handle_attribute(self, atom, changes):
        subj      = atom["subject"]
        subj_type = atom["subject_type"]
        obj       = atom["object"]
        obj_type  = atom.get("object_type", "CONCEPT")

        self._ensure_node(subj, subj_type, changes)

        if subj_type == "PERSON":
            self._last_subject = subj.lower()

        # Identity revelation: "nikhil:mother is Sunita"
        # subject is a placeholder (contains ":") and object is a PERSON → merge
        if ":" in subj and obj_type == "PERSON":
            real_name = obj.lower()
            self._ensure_node(real_name, "PERSON", changes)
            self.graph.merge_nodes(subj, real_name, "PERSON")
            changes["new_edges"].append(f"resolved: {subj} → {real_name}")
            return

        self.graph.add_attribute(subj, obj)
        changes["attributes"].append(f"{subj} → [{obj}]")

    def _handle_state(self, atom, changes):
        subj      = atom["subject"]
        subj_type = atom["subject_type"]
        state     = atom["state"]
        date      = atom["date"]

        self._ensure_node(subj, subj_type, changes)

        if subj_type == "PERSON":
            self._last_subject = subj.lower()

        self.graph.add_state(subj, state, date)
        changes["states"].append(f"{subj} --[IS | {date}]--> {state}")

    def _handle_modal_category(self, atom, changes):
        subj      = atom["subject"]
        obj       = atom["object"]
        subj_type = atom["subject_type"]
        obj_type  = atom["object_type"]
        modal     = atom.get("modal", "could")
        date      = atom["date"]

        self._ensure_node(subj, subj_type, changes)
        self._ensure_node(obj,  obj_type,  changes)

        edge, _ = self.graph.add_edge(subj, "COULD_BE", obj, date, confidence=0.7)
        changes["inferred"].append(f"'{subj}' --[{modal} be]--> '{obj}'")

    def _handle_possessive(self, atom, changes):
        owner        = atom["owner"]
        owner_type   = atom["owner_type"]
        relation     = atom["relation"]
        entity       = atom["entity"]
        entity_type  = atom["entity_type"]
        entity_named = atom["entity_named"]
        date         = atom["date"]

        self._ensure_node(owner,  owner_type,  changes)
        self._ensure_node(entity, entity_type, changes)

        if owner_type == "PERSON":
            self._last_subject = owner.lower()

        edge, superseded_ids = self.graph.add_edge(owner, relation, entity, date)
        changes["new_edges"].append(f"{owner} --[{relation} | {date}]--> {entity}")

        for eid in superseded_ids:
            e = self.graph.edge_store[eid]
            changes["superseded"].append(f"{e.subject} --[{e.predicate}]--> {e.object}")

        # If a real name was given (e.g. "Nikhil's mother Sunita"),
        # check if the placeholder already exists and merge it into the real node.
        if entity_named:
            rel_word    = next(
                (k for k, v in POSSESSIVE_PREDICATES.items() if v == relation), None
            )
            placeholder = f"{owner.lower()}:{rel_word}" if rel_word else None
            if placeholder and self.graph.get_node(placeholder):
                self.graph.merge_nodes(placeholder, entity, entity_type)
                changes["new_edges"].append(f"resolved: {placeholder} → {entity}")

    # ── snapshot for visualizer / dump ────────────────────────────

    def snapshot(self):
        return self.graph.snapshot()
