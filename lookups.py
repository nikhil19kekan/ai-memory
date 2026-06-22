"""
Deterministic lookup tables.
These replace LLM calls for known patterns.
Personal knowledge only — no world knowledge stored here.
"""

# ── verb lemma → edge predicate ──────────────────────────────────
VERB_EDGE_MAP = {
    # positive affect
    "like":    "LIKES",
    "love":    "LOVES",
    "enjoy":   "LIKES",
    "adore":   "LOVES",
    "prefer":  "PREFERS",
    "crave":   "CRAVES",
    "want":    "WANTS",
    "need":    "NEEDS",
    "fancy":   "LIKES",
    "relish":  "LIKES",
    # negative affect
    "hate":    "HATES",
    "dislike": "DISLIKES",
    "despise": "HATES",
    "avoid":   "AVOIDS",
    "detest":  "HATES",
    "loathe":  "HATES",
    # actions
    "cook":    "COOKS",
    "eat":     "EATS",
    "drink":   "DRINKS",
    "make":    "MAKES",
    "prepare": "COOKS",
    "work":    "WORKS_AT",
    "know":    "KNOWS",
    "own":     "OWNS",
    "have":    "HAS",
    "visit":   "VISITS",
    "read":    "READS",
    "watch":   "WATCHES",
    "play":    "PLAYS",
    "study":   "STUDIES",
    "use":     "USES",
    "buy":     "BUYS",
    "live":    "LIVES_IN",
    "meet":    "MEETS",
    "run":     "RUNS",
    "drive":   "DRIVES",
    "listen":  "LISTENS_TO",
    "follow":  "FOLLOWS",
    "support": "SUPPORTS",
    # organizational roles (used by supersession cascade)
    "lead":    "LEADS",
    "manage":  "MANAGES",
    "head":    "LEADS",
    "join":    "WORKS_AT",
}

# ── NER labels that indicate a token is a quantity, not an object ──
QUANTITY_NER_LABELS = {"CARDINAL", "QUANTITY", "PERCENT", "ORDINAL", "DATE", "TIME"}

# ── verbs where a missing object implies a default ────────────────
IMPLIED_OBJECTS = {
    "COOKS":  "food",
    "EATS":   "food",
    "DRINKS": "drink",
}

# ── temporary state adjectives (get dated, not stored as attributes) ──
STATE_ADJECTIVES = {
    "sick", "stressed", "tired", "happy", "sad", "angry", "busy",
    "free", "hungry", "full", "bored", "excited", "anxious", "relaxed",
    "ill", "well", "unwell", "upset", "content", "frustrated", "confused",
}

# ── possessive relation words → graph predicates ──────────────────
# Used to parse "Nikhil's mother lives in Delhi" → nikhil MOTHER nikhil:mother
POSSESSIVE_PREDICATES = {
    "mother":      "MOTHER",
    "father":      "FATHER",
    "wife":        "WIFE",
    "husband":     "HUSBAND",
    "son":         "SON",
    "daughter":    "DAUGHTER",
    "brother":     "BROTHER",
    "sister":      "SISTER",
    "friend":      "FRIEND",
    "boss":        "BOSS",
    "colleague":   "COLLEAGUE",
    "partner":     "PARTNER",
    "girlfriend":  "GIRLFRIEND",
    "boyfriend":   "BOYFRIEND",
    "grandfather": "GRANDFATHER",
    "grandmother": "GRANDMOTHER",
    "uncle":       "UNCLE",
    "aunt":        "AUNT",
    "cousin":      "COUSIN",
    "nephew":      "NEPHEW",
    "niece":       "NIECE",
    "mentor":      "MENTOR",
    "manager":     "MANAGER",
    "teacher":     "TEACHER",
    "coach":       "COACH",
}

# ── supersession cascade rules ────────────────────────────────────
# When a WORKS_AT edge is superseded, flag related role-at-same-org edges.
# When a LIVES_IN edge is superseded, no cascades needed.
SUPERSESSION_CASCADE = {
    "WORKS_AT": ["LEADS", "MANAGES", "REPORTS_TO", "WORKS_WITH"],
    "LIVES_IN": [],
}
