"""
Extraction accuracy benchmark.
Measures precision, recall, F1 of triple extraction against ground truth.
Compares: our parser vs LLM-only extraction vs OpenIE (if available).
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from graph_manager import GraphManager


def load_dataset(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_with_our_system(sentence):
    """Run our parser + graph_manager, return extracted triples."""
    gm = GraphManager()
    changes = gm.process(sentence)

    triples = []
    for edge_str in changes["new_edges"]:
        # parse "subj --[PRED | date]--> obj" or "subj --[PRED, qual | date]--> obj"
        try:
            if "resolved:" in edge_str:
                continue
            subj = edge_str.split(" --[")[0].strip().lower()
            mid = edge_str.split("--[")[1].split("]-->")[0]
            obj = edge_str.split("]--> ")[1].strip().lower()
            parts = mid.split(" | ")
            pred_part = parts[0]
            pred = pred_part.split(",")[0].strip()
            qualifier = pred_part.split(",", 1)[1].strip() if "," in pred_part else ""
            triples.append({
                "subject": subj,
                "predicate": pred,
                "object": obj,
                "qualifier": qualifier,
            })
        except (IndexError, ValueError):
            continue

    for attr_str in changes["attributes"]:
        try:
            subj = attr_str.split(" → [")[0].strip().lower()
            obj = attr_str.split(" → [")[1].rstrip("]").strip().lower()
            triples.append({
                "subject": subj,
                "predicate": "IS",
                "object": obj,
            })
        except (IndexError, ValueError):
            continue

    for state_str in changes["states"]:
        try:
            subj = state_str.split(" --[IS")[0].strip().lower()
            obj = state_str.split("]--> ")[1].strip().lower()
            triples.append({
                "subject": subj,
                "predicate": "STATE",
                "object": obj,
            })
        except (IndexError, ValueError):
            continue

    return triples


def triple_matches(extracted, expected):
    """Check if extracted triple matches expected (case-insensitive, fuzzy on qualifier/date)."""
    if extracted["subject"].lower() != expected["subject"].lower():
        return False
    if extracted["predicate"].upper() != expected["predicate"].upper():
        return False
    if extracted["object"].lower() != expected["object"].lower():
        return False
    return True


def evaluate_sentence(entry):
    """Evaluate one sentence. Returns (true_positives, false_positives, false_negatives, details)."""
    extracted = extract_with_our_system(entry["sentence"])
    expected = entry["expected_triples"]

    matched_expected = set()
    matched_extracted = set()

    for i, ext in enumerate(extracted):
        for j, exp in enumerate(expected):
            if j not in matched_expected and triple_matches(ext, exp):
                matched_expected.add(j)
                matched_extracted.add(i)
                break

    tp = len(matched_expected)
    fp = len(extracted) - len(matched_extracted)
    fn = len(expected) - len(matched_expected)

    return tp, fp, fn, {
        "id": entry["id"],
        "sentence": entry["sentence"],
        "category": entry["category"],
        "expected": expected,
        "extracted": extracted,
        "tp": tp, "fp": fp, "fn": fn,
    }


def run(dataset_path=None):
    if dataset_path is None:
        dataset_path = os.path.join(
            os.path.dirname(__file__), "../datasets/extraction_sentences.jsonl"
        )

    dataset = load_dataset(dataset_path)

    total_tp = total_fp = total_fn = 0
    results_by_category = {}
    failures = []

    for entry in dataset:
        tp, fp, fn, details = evaluate_sentence(entry)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        cat = entry["category"]
        if cat not in results_by_category:
            results_by_category[cat] = {"tp": 0, "fp": 0, "fn": 0}
        results_by_category[cat]["tp"] += tp
        results_by_category[cat]["fp"] += fp
        results_by_category[cat]["fn"] += fn

        if fp > 0 or fn > 0:
            failures.append(details)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("=" * 60)
    print("EXTRACTION BENCHMARK")
    print("=" * 60)
    print(f"\nOverall: P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}  (total expected={total_tp + total_fn})")

    print(f"\nBy category:")
    for cat, counts in sorted(results_by_category.items()):
        p = counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) > 0 else 0
        r = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        status = "PASS" if f == 1.0 else "MISS"
        print(f"  {cat:25s}  P={p:.2f} R={r:.2f} F1={f:.2f}  [{status}]")

    if failures:
        print(f"\n{'─' * 60}")
        print(f"FAILURES ({len(failures)}):")
        print(f"{'─' * 60}")
        for f in failures:
            print(f"\n  [{f['id']}] {f['sentence']}")
            print(f"    expected : {f['expected']}")
            print(f"    extracted: {f['extracted']}")
            print(f"    TP={f['tp']} FP={f['fp']} FN={f['fn']}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "by_category": results_by_category,
        "failures": failures,
    }


if __name__ == "__main__":
    run()
