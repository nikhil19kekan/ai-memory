"""
Self-improving rule cache — learned from LLM calls, persisted to disk.
Checked before making LLM calls to avoid redundant API requests.

Two rule types:
  1. verb_map:       verb_lemma → PREDICATE (extends VERB_EDGE_MAP at runtime)
  2. pattern_rules:  structural patterns like "into" + noun → INTERESTED_IN
"""

import json
import os

_RULES_FILE = os.path.join(os.path.dirname(__file__), "learned_rules.json")


class LearnedRules:
    def __init__(self, path=None):
        self.path = path or _RULES_FILE
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                data = json.load(f)
        else:
            data = {}
        self.verb_map = data.get("verb_map", {})
        self.pattern_rules = data.get("pattern_rules", [])

    def save(self):
        with open(self.path, "w") as f:
            json.dump({
                "verb_map": self.verb_map,
                "pattern_rules": self.pattern_rules,
            }, f, indent=2)

    # ── verb map ──────────────────────────────────────────────────

    def get_verb_predicate(self, lemma: str):
        """Look up a learned verb→predicate mapping. Returns None if not found."""
        return self.verb_map.get(lemma.lower())

    def add_verb_mapping(self, lemma: str, predicate: str):
        key = lemma.lower()
        pred = predicate.upper()
        if self.verb_map.get(key) == pred:
            return
        self.verb_map[key] = pred
        self.save()

    # ── pattern rules ─────────────────────────────────────────────

    def get_prep_predicate(self, prep: str):
        """Look up a learned preposition→predicate pattern. Returns None if not found."""
        for rule in self.pattern_rules:
            if rule.get("type") == "prep_pattern" and rule.get("prep") == prep.lower():
                return rule["predicate"]
        return None

    def add_pattern_rule(self, rule: dict):
        for existing in self.pattern_rules:
            if (existing.get("type") == rule.get("type")
                    and existing.get("prep", "") == rule.get("prep", "")
                    and existing.get("trigger", "") == rule.get("trigger", "")):
                return  # already exists
        self.pattern_rules.append(rule)
        self.save()


# ── module-level singleton ────────────────────────────────────────

_instance = None


def get_rules(path=None) -> LearnedRules:
    global _instance
    if _instance is None:
        _instance = LearnedRules(path)
    return _instance
