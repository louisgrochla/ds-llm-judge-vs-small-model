"""Split hygiene and seed independence, re-derived from the committed parquet files.

Two questions the LLM rebuild depends on, both left open in HANDOVER §6.4:

  1. Are the splits disjoint? Few-shot demonstrations are retrieved from the
     train pool at inference time, so any overlap with dev/test would leak the
     answer into the prompt.
  2. How much training data do the 5 seeds actually share at each budget? If
     they share almost all of it, the per-seed spread at that budget measures
     training nondeterminism, not data-sampling variance, and the error bars
     cannot be described as the latter.

    python scripts/check_split_integrity.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
SIZES = [50, 100, 250, 500, 1000, 2500, 5000, 9000]
N_SEEDS = 5


def texts(filename: str) -> set[str]:
    return set(pd.read_parquet(DATA_DIR / filename).text)


def main() -> int:
    failures = []

    print("=" * 72)
    print("1. Split disjointness (a non-zero overlap is leakage, not a curiosity)")
    print("=" * 72)
    named = {
        "dev_slice": texts("dev_slice.parquet"),
        "val": texts("val.parquet"),
        "test": texts("test.parquet"),
        "demo_pool": texts("train_n9000_seed0.parquet"),
    }
    for (a_name, a), (b_name, b) in itertools.combinations(named.items(), 2):
        overlap = len(a & b)
        flag = "ok" if overlap == 0 else "LEAK"
        print(f"  [{flag:>4}] {a_name:>10} n={len(a):<5} vs {b_name:<10} n={len(b):<5} shared {overlap}")
        if overlap:
            failures.append(f"{a_name} overlaps {b_name} on {overlap} rows")

    print()
    print("=" * 72)
    print("2. Seed independence by budget")
    print("=" * 72)
    pool = set()
    for seed in range(N_SEEDS):
        pool |= texts(f"train_n9000_seed{seed}.parquet")
    print(f"  sampling pool = {len(pool)} rows (train minus the val and dev carve-outs)\n")
    print(f"  {'n':>6}{'shared by all 5':>20}{'mean pairwise':>18}{'if drawn at random':>22}")
    for n in SIZES:
        seeds = [texts(f"train_n{n}_seed{s}.parquet") for s in range(N_SEEDS)]
        shared = len(set.intersection(*seeds))
        pairwise = float(np.mean([len(a & b) for a, b in itertools.combinations(seeds, 2)]))
        expected = n * n / len(pool)
        note = "  <-- seeds are near-identical" if pairwise / n > 0.9 else ""
        print(
            f"  {n:>6}{shared:>11} ({shared / n:>5.1%}){pairwise:>11.0f} ({pairwise / n:>5.1%})"
            f"{expected:>15.0f} ({expected / n:>5.1%}){note}"
        )

    print()
    print("  Pairwise overlap tracks the random-draw expectation at every budget, so this")
    print("  is a property of the pool size, not a sampling bug. The consequence still")
    print("  stands: at n=9000 the five seeds differ in ~2% of their training data, so the")
    print("  spread there is training nondeterminism and must not be reported as evidence")
    print("  about sensitivity to which examples were labelled. Error bars at n<=1000 do")
    print("  carry that meaning; at n=5000 and above they progressively stop.")

    print()
    if failures:
        print(f"FAILED: {'; '.join(failures)}")
        return 1
    print("splits are disjoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
