"""
Rule cache convergence benchmark.
Measures how quickly LLM calls decrease as the system learns rules.

Feeds diverse sentences and tracks:
  - LLM calls per sentence (should converge toward 0)
  - Cumulative rules learned
  - Cost savings vs always-call-LLM baseline
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Sentences using progressively repeated patterns.
# After the LLM handles a pattern once, subsequent uses should be cached.
CONVERGENCE_SENTENCES = [
    # Round 1: novel patterns → LLM calls expected
    ("Nikhil does yoga", True, "do → PRACTICES"),
    ("Nikhil is into running", True, "into → INTERESTED_IN"),
    ("Nikhil admires Elon Musk", True, "admire → ADMIRES"),

    # Round 2: same patterns, different words → should use cached rules
    ("Priya does karate", False, "do (cached)"),
    ("Ravi is into cooking", False, "into (cached)"),
    ("Sunita admires Gandhi", False, "admire (cached)"),

    # Round 3: known verbs → never needed LLM
    ("Nikhil likes pizza", False, "like (static)"),
    ("Priya hates mushrooms", False, "hate (static)"),
    ("Ravi lives in Delhi", False, "live (static)"),

    # Round 4: more novel patterns → LLM
    ("Nikhil explores machine learning", True, "explore → EXPLORES"),
    ("Priya mentors junior doctors", True, "mentor → MENTORS"),

    # Round 5: reuse newly learned
    ("Ravi explores quantum computing", False, "explore (cached)"),
    ("Sunita mentors young teachers", False, "mentor (cached)"),

    # Round 6: still more novel
    ("Nikhil collects vinyl records", True, "collect → COLLECTS"),
    ("Priya teaches anatomy", True, "teach → TEACHES"),

    # Round 7: reuse
    ("Ravi collects stamps", False, "collect (cached)"),
    ("Nikhil teaches Python", False, "teach (cached)"),

    # Round 8: all known by now — zero LLM calls expected
    ("Sunita does meditation", False, "do (cached)"),
    ("Ravi admires Nikhil", False, "admire (cached)"),
    ("Priya is into gardening", False, "into (cached)"),
    ("Nikhil explores robotics", False, "explore (cached)"),
]


def run():
    from graph_manager import GraphManager
    from learned_rules import get_rules, _instance

    # Reset learned rules for clean measurement
    import learned_rules as lr
    lr._instance = None
    rules_path = os.path.join(os.path.dirname(__file__), "../../learned_rules.json")
    if os.path.exists(rules_path):
        os.rename(rules_path, rules_path + ".backup")

    gm = GraphManager()
    rules = get_rules()

    results = []
    total_llm_calls = 0
    total_cached = 0
    has_api = bool(os.environ.get("GEMINI_API_KEY", ""))

    # Check config.json for API key
    config_path = os.path.join(os.path.dirname(__file__), "../../config.json")
    if not has_api and os.path.exists(config_path):
        with open(config_path) as f:
            has_api = bool(json.load(f).get("gemini_api_key", ""))

    print("=" * 70)
    print("RULE CACHE CONVERGENCE BENCHMARK")
    print("=" * 70)

    if not has_api:
        print("\n  ⚠ No API key configured. Running in dry-run mode.")
        print("  Set GEMINI_API_KEY or add to config.json for live results.")
        print("  Showing expected behavior based on system design.\n")

    print(f"\n  {'#':<4} {'Sentence':<45} {'LLM?':<8} {'Rules':<6} {'Note'}")
    print(f"  {'─' * 80}")

    for i, (sentence, expected_llm, note) in enumerate(CONVERGENCE_SENTENCES):
        rules_before = len(rules.verb_map) + len(rules.pattern_rules)

        changes = gm.process(sentence)

        rules_after = len(rules.verb_map) + len(rules.pattern_rules)
        used_llm = len(changes.get("inferred", [])) > 0 and any(
            "learned rule:" in inf for inf in changes.get("inferred", [])
        )
        had_unknown = len(changes.get("unknown_verbs", [])) > 0

        if used_llm:
            total_llm_calls += 1
            llm_marker = "YES"
        elif had_unknown:
            llm_marker = "FAIL"  # should have called LLM but couldn't
        else:
            total_cached += 1
            llm_marker = "no"

        match = "✓" if (used_llm == expected_llm) or (not had_unknown and not expected_llm) else "✗"
        print(f"  {i+1:<4} {sentence:<45} {llm_marker:<8} {rules_after:<6} {note} {match}")

        results.append({
            "index": i + 1,
            "sentence": sentence,
            "used_llm": used_llm,
            "expected_llm": expected_llm,
            "rules_count": rules_after,
            "match": match == "✓",
        })

        if has_api and used_llm:
            time.sleep(1)  # respect rate limits

    total = len(CONVERGENCE_SENTENCES)
    print(f"\n  {'─' * 80}")
    print(f"  Total sentences:    {total}")
    print(f"  LLM calls:          {total_llm_calls}")
    print(f"  Cached/static:      {total_cached}")
    print(f"  LLM call rate:      {total_llm_calls/total:.1%}")
    print(f"  Cache hit rate:     {total_cached/total:.1%}")
    print(f"  Rules learned:      {len(rules.verb_map)} verbs + {len(rules.pattern_rules)} patterns")

    if total_llm_calls > 0:
        # always-LLM baseline would call on every non-static-verb sentence
        non_static = sum(1 for _, exp, _ in CONVERGENCE_SENTENCES if exp or "cached" in _)
        savings = (1 - total_llm_calls / max(non_static, 1)) * 100
        print(f"\n  vs always-call-LLM baseline:")
        print(f"    Baseline would make:  {non_static} calls")
        print(f"    We made:              {total_llm_calls} calls")
        print(f"    Savings:              {savings:.0f}%")

    # Restore backup
    if os.path.exists(rules_path + ".backup"):
        if os.path.exists(rules_path):
            os.remove(rules_path)
        os.rename(rules_path + ".backup", rules_path)

    return results


if __name__ == "__main__":
    run()
