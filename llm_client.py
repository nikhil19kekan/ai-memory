"""
LLM fallback for sentences the deterministic parser cannot handle.
Uses Gemini Flash (free tier) to parse the sentence AND return generalizable rules.
Rules are cached by learned_rules.py so the same pattern never needs an LLM call again.
"""

import json
import os
import logging

_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a knowledge graph parser. Given a sentence about a person, extract structured facts and return generalizable rules for future parsing.

Return ONLY valid JSON with two keys:

1. "atoms" — list of facts extracted from the sentence:
   - "subject": entity performing the action (lowercase)
   - "subject_type": PERSON | PLACE | ORG | CONCEPT
   - "predicate": relationship in UPPER_SNAKE_CASE (e.g., PRACTICES, INTERESTED_IN, ADMIRES, SKILLED_AT)
   - "object": target entity (lowercase)
   - "object_type": PERSON | PLACE | ORG | CONCEPT
   - "qualifier": adverb/frequency if present, else ""
   - "negated": boolean

2. "learned" — generalizable rules to cache so identical patterns never need an LLM call again:
   - "verb_map": dict of {verb_lemma: "PREDICATE"} for any new verb-to-predicate mappings.
     These extend the parser's verb lookup table. Use the base verb form (lemma).
     NEVER include common auxiliary/linking verbs: be, have, do, will, shall, can, may, must, would, could, should.
     Only include content verbs with clear semantic meaning (e.g., admire, practice, explore).
     Example: {"admire": "ADMIRES", "explore": "EXPLORES"}
   - "pattern_rules": list of structural patterns (non-verb-based), each with:
     - "type": "prep_pattern"
     - "prep": the preposition that triggers this pattern (lowercase)
     - "predicate": the PREDICATE to assign
     - "description": one-line explanation of when this rule applies

Rules must be GENERAL — they apply to future sentences with the same structure:
  - "do" + activity → PRACTICES (yoga, karate, meditation, etc.)
  - "into" + noun → INTERESTED_IN (running, cooking, art, etc.)
  - "admire" → ADMIRES (any object)
  - "fan of" + noun → could be a prep_pattern with prep "of" if triggered by "fan"

Keep predicates consistent: use existing ones where possible (LIKES, LOVES, HATES, PRACTICES, INTERESTED_IN, ADMIRES, SKILLED_AT, etc.)."""


def _get_api_key() -> str:
    """Load API key from config.json, falling back to env var."""
    if os.path.exists(_CONFIG_FILE):
        try:
            with open(_CONFIG_FILE) as f:
                key = json.load(f).get("gemini_api_key", "")
            if key:
                return key
        except (json.JSONDecodeError, OSError):
            pass
    return os.environ.get("GEMINI_API_KEY", "")


def parse_with_llm(sentence: str, subject: str = None, date: str = "") -> tuple[list[dict], dict]:
    """
    Send sentence to Gemini Flash for parsing.

    Returns:
        (atoms, learned_rules)
        atoms:         list of verb_relation-compatible dicts
        learned_rules: {"verb_map": {...}, "pattern_rules": [...]}
        Returns ([], {}) if no API key or on error.
    """
    api_key = _get_api_key()
    if not api_key:
        return [], {}

    try:
        from google import genai
    except ImportError:
        return [], {}

    client = genai.Client(api_key=api_key)

    user_msg = f'Parse this sentence: "{sentence}"'
    if subject:
        user_msg += f"\nThe main subject/person is: {subject}"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=user_msg,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "response_mime_type": "application/json",
            },
        )
        text = response.text
    except Exception as e:
        log.warning(f"LLM call failed: {e}")
        return [], {}

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, IndexError):
        log.warning(f"LLM returned invalid JSON: {text[:200]}")
        return [], {}

    atoms = []
    for item in data.get("atoms", []):
        predicate = item.get("predicate", "")
        obj = item.get("object", "")
        if not predicate or not obj:
            continue
        atoms.append({
            "type":         "verb_relation",
            "subject":      item.get("subject", subject or ""),
            "subject_type": item.get("subject_type", "PERSON"),
            "edge_type":    predicate.upper(),
            "object":       obj,
            "object_type":  item.get("object_type", "CONCEPT"),
            "qualifier":    item.get("qualifier", ""),
            "negated":      item.get("negated", False),
            "date":         date,
        })

    learned = data.get("learned", {})
    return atoms, learned
