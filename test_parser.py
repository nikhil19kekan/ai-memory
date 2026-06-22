"""
Comprehensive parser test — no matplotlib, no interactive loop.
Runs a battery of English sentence types and prints what the parser extracts.
"""

import sys
sys.path.insert(0, "/Users/nik/projects/ai-memory/knowledge_graph")

from graph_manager import GraphManager

TESTS = [
    # ── simple verb relations ─────────────────────────────────────
    ("01 simple like",           "Nikhil likes Indian food"),
    ("02 simple love",           "Nikhil loves pizza"),
    ("03 simple hate",           "Nikhil hates broccoli"),
    ("04 simple dislike",        "Nikhil dislikes coffee"),
    ("05 negated like",          "Nikhil does not like sushi"),
    ("06 negated hate",          "Nikhil does not hate pasta"),
    # ── adverb qualifiers ─────────────────────────────────────────
    ("07 qualifier a lot",       "Nikhil likes biryani a lot"),
    ("08 qualifier sometimes",   "Nikhil sometimes eats pizza"),
    ("09 qualifier half times",  "Nikhil eats curry half of the times"),
    # ── compound / conjunct objects ───────────────────────────────
    ("10 conj objects",          "Nikhil likes pizza and sushi"),
    ("11 conj objects 3",        "Nikhil loves biryani, dosa and samosa"),
    # ── action verbs ─────────────────────────────────────────────
    ("12 cooks",                 "Nikhil cooks"),
    ("13 cooks object",          "Nikhil cooks pasta"),
    ("14 eats",                  "Nikhil eats food"),
    ("15 works",                 "Nikhil works at Google"),
    ("16 lives in",              "Nikhil lives in Mumbai"),
    ("17 visits",                "Nikhil visits Japan"),
    # ── linking verb / attributes ─────────────────────────────────
    ("18 simple attribute",      "Nikhil is a software engineer"),
    ("19 multi attribute",       "Nikhil is a software engineer and a food lover"),
    ("20 complex attribute",     "Nikhil is a 35 years old indian male, a software engineer and a food lover"),
    ("21 attribute + relcl",     "Nikhil is a male who loves Indian food"),
    ("22 attribute + relcl neg", "Nikhil is a person who does not like broccoli"),
    # ── state adjectives (temporal) ───────────────────────────────
    ("23 state stressed",        "Nikhil is stressed"),
    ("24 state happy",           "Nikhil is happy"),
    ("25 state sick",            "Nikhil is sick"),
    ("26 state hungry",          "Nikhil is hungry"),
    # ── modal (timeless structural) ───────────────────────────────
    ("27 modal could be",        "Biryani could be Indian food"),
    ("28 modal might be",        "Samosa might be Indian food"),
    # ── pronoun coreference ───────────────────────────────────────
    ("29 pronoun he",            "He loves dosa"),        # after Nikhil
    ("30 pronoun she",           "Priya is a chef"),
    ("31 pronoun she ref",       "She loves sushi"),      # after Priya
    # ── past tense ───────────────────────────────────────────────
    ("32 past tense liked",      "Nikhil liked pizza yesterday"),
    ("33 past tense hated",      "Nikhil hated coffee last year"),
    # ── supersession ─────────────────────────────────────────────
    ("34 supersede: now loves",  "Nikhil now loves coffee"),  # previously hated
    # ── place / org entities ─────────────────────────────────────
    ("35 place",                 "Nikhil lives in Bangalore"),
    ("36 org",                   "Nikhil works at Microsoft"),
    # ── unknown verb (LLM path) ───────────────────────────────────
    ("37 unknown verb",          "Nikhil admires Elon Musk"),
]


def run():
    gm = GraphManager()
    passed = 0
    failed = 0

    for label, sentence in TESTS:
        print(f"\n{'─'*60}")
        print(f"[{label}]  {sentence!r}")
        try:
            changes = gm.process(sentence)
            if changes["new_entities"]:
                print(f"  entities : {', '.join(changes['new_entities'])}")
            for e in changes["new_edges"]:
                print(f"  edge     : {e}")
            for a in changes["attributes"]:
                print(f"  info     : {a}")
            for s in changes["states"]:
                print(f"  state    : {s}")
            for i in changes["inferred"]:
                print(f"  inferred : {i}")
            if changes["superseded"]:
                print(f"  supersed : {', '.join(changes['superseded'])}")
            if changes["unknown_verbs"]:
                print(f"  UNKNOWN  : {', '.join(changes['unknown_verbs'])}")
            if not any([changes["new_edges"], changes["attributes"],
                        changes["states"], changes["inferred"],
                        changes["unknown_verbs"]]):
                print(f"  !! NO OUTPUT — parser produced nothing")
                failed += 1
            else:
                passed += 1
        except Exception as exc:
            import traceback
            print(f"  EXCEPTION: {exc}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} had no output or crashed")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()
