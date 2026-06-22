"""
Run all benchmarks and produce a summary report.

Usage:
  python3 eval/run_benchmarks.py              # run all
  python3 eval/run_benchmarks.py extraction    # run one
  python3 eval/run_benchmarks.py retrieval
  python3 eval/run_benchmarks.py convergence
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metrics import extraction_eval, retrieval_eval, convergence_eval


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["extraction", "retrieval", "convergence"]

    results = {}

    if "extraction" in targets:
        print("\n")
        results["extraction"] = extraction_eval.run()

    if "retrieval" in targets:
        print("\n")
        results["retrieval"] = retrieval_eval.run()

    if "convergence" in targets:
        print("\n")
        results["convergence"] = convergence_eval.run()

    # Summary
    print("\n")
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    if "extraction" in results:
        e = results["extraction"]
        print(f"\n  Extraction:  P={e['precision']:.3f}  R={e['recall']:.3f}  F1={e['f1']:.3f}")

    if "retrieval" in results:
        r = results["retrieval"]
        print(f"\n  Retrieval (entity recall):")
        print(f"    Graph:   {r['graph_avg_recall']:.3f}")
        print(f"    Vector:  {r['vector_avg_recall']:.3f}")
        print(f"    Delta:   {r['graph_avg_recall'] - r['vector_avg_recall']:+.3f}")

    if "convergence" in results:
        c = results["convergence"]
        total = len(c)
        llm_calls = sum(1 for r in c if r["used_llm"])
        cached = sum(1 for r in c if not r["used_llm"] and r["match"])
        print(f"\n  Convergence:")
        print(f"    LLM calls: {llm_calls}/{total} ({llm_calls/total:.0%})")
        print(f"    Cached:    {cached}/{total}")


if __name__ == "__main__":
    main()
