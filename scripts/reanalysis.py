"""Re-analysis of the label-budget sweep (HANDOVER §6.4). No API key required.

Produces the four things the published analysis got wrong or never computed:

  1. A real p-value, via paired approximate randomisation (the repo claimed one it
     never computed).
  2. Uncertainty over training seeds as well as test rows, shown side by side with
     the rows-only interval so the omitted uncertainty is visible rather than argued.
  3. A confidence interval on the crossover budget itself, which is the quantity the
     headline claim is actually about.
  4. The label budget with the fixed 500-row validation set counted, since it is
     supervision and the published x-axis excluded it.

Plus the comparison the study never ran: fine-tuned DistilBERT against frozen-encoder
and bag-of-character-n-gram baselines at every budget.

    python scripts/reanalysis.py                       # encoder arms only
    python scripts/reanalysis.py --baseline <parquet>  # add an LLM arm once rebuilt

Passing the withdrawn v1 LLM predictions is possible but deliberately loud: that file
was collected with the test set ordered by class and nothing derived from it should be
quoted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stats import (  # noqa: E402
    crossover_interval,
    hierarchical_bootstrap,
    label_budget,
    macro_f1,
    paired_permutation_test,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
FT_DIR = ROOT / "results" / "finetune" / "v2_retune"
FROZEN_DIR = ROOT / "results" / "frozen_encoder"
OUT = ROOT / "results" / "reanalysis_summary.json"
SIZES = [50, 100, 250, 500, 1000, 2500, 5000, 9000]
WITHDRAWN = "llm_predictions_v1_test.parquet"


def load_finetune_probs(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (per-seed probs, true labels) for one budget, in on-disk row order."""
    files = sorted(FT_DIR.glob(f"n{n}_seed*.parquet"))
    if not files:
        raise FileNotFoundError(f"no fine-tune predictions for n={n} in {FT_DIR}")
    frames = [pd.read_parquet(p) for p in files]
    probs = np.stack([np.array(f.probs.tolist(), dtype=np.float32) for f in frames])
    return probs, frames[0].true_label.to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=None,
                    help="Parquet of LLM predictions with columns [text, pred_intent].")
    ap.add_argument("--permutations", type=int, default=10_000)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    names = (DATA_DIR / "label_names.txt").read_text().splitlines()
    n_classes = len(names)
    name_to_id = {name: i for i, name in enumerate(names)}

    ft_probs, y_true = {}, None
    for n in SIZES:
        probs, truth = load_finetune_probs(n)
        ft_probs[n] = probs
        if y_true is None:
            y_true = truth
        elif not np.array_equal(y_true, truth):
            raise ValueError(f"row order differs between budgets at n={n}")

    summary: dict = {
        "n_classes": n_classes,
        "n_test_rows": int(len(y_true)),
        "note": "Uncertainty is over test rows AND training seeds unless stated otherwise.",
    }

    # -- 1. label-budget-corrected curve -----------------------------------
    print("=" * 78)
    print("1. Quality by budget, with the validation set counted as supervision")
    print("=" * 78)
    print(f"  {'n':>6}{'true budget':>13}{'single-seed':>13}{'ensemble':>11}")
    curve = {}
    for n in SIZES:
        per_seed = [macro_f1(y_true, ft_probs[n][s].argmax(1), n_classes)
                    for s in range(ft_probs[n].shape[0])]
        ens = macro_f1(y_true, ft_probs[n].mean(0).argmax(1), n_classes)
        curve[n] = {
            "label_budget_including_val": label_budget(n),
            "single_seed_mean": round(float(np.mean(per_seed)), 4),
            "single_seed_sd": round(float(np.std(per_seed)), 4),
            "ensemble": round(float(ens), 4),
        }
        print(f"  {n:>6}{label_budget(n):>13}{np.mean(per_seed):>13.4f}{ens:>11.4f}")
    summary["finetuned_by_budget"] = curve
    print("\n  At n=50 the published x-axis understated the true label cost by 11x.")

    # -- 2. frozen-encoder arms -------------------------------------------
    frozen = {}
    if FROZEN_DIR.exists():
        encoders = sorted({p.name.split("_")[1] for p in FROZEN_DIR.glob("probs_*_n*.npy")
                           if not p.name.startswith("probs_perseed")})
        for enc in encoders:
            rows = {}
            for n in SIZES:
                path = FROZEN_DIR / f"probs_{enc}_n{n}.npy"
                if not path.exists():
                    continue
                probs = np.load(path)
                gap = macro_f1(y_true, probs.argmax(1), n_classes) - macro_f1(
                    y_true, ft_probs[n].mean(0).argmax(1), n_classes)
                test = paired_permutation_test(
                    y_true, probs.argmax(1), ft_probs[n].mean(0).argmax(1),
                    n_classes, n_permutations=args.permutations, seed=42)
                rows[n] = {
                    "ensemble_macro_f1": round(float(macro_f1(y_true, probs.argmax(1), n_classes)), 4),
                    "gap_vs_finetuned_ensemble": round(float(gap), 4),
                    "p_value": round(test["p_value"], 5),
                }
            frozen[enc] = rows
        if frozen:
            print()
            print("=" * 78)
            print("2. Frozen baselines vs the fine-tuned ensemble (paired randomisation)")
            print("=" * 78)
            for enc, rows in frozen.items():
                print(f"\n  {enc}")
                print(f"  {'n':>6}{'frozen':>10}{'fine-tuned':>13}{'gap':>10}{'p':>10}")
                for n, r in rows.items():
                    ft = curve[n]["ensemble"]
                    print(f"  {n:>6}{r['ensemble_macro_f1']:>10.4f}{ft:>13.4f}"
                          f"{r['gap_vs_finetuned_ensemble']:>+10.4f}{r['p_value']:>10.4f}")
    summary["frozen_baselines"] = frozen

    # -- 3. uncertainty, with and without seed resampling ------------------
    baseline_pred = None
    if args.baseline:
        path = Path(args.baseline)
        if path.name == WITHDRAWN:
            print("\n" + "!" * 78)
            print("!! Using results/" + WITHDRAWN + " as the baseline.")
            print("!! That file was collected with the test set ORDERED BY CLASS. Every")
            print("!! number below that references it is invalid and must not be quoted.")
            print("!" * 78)
        llm = pd.read_parquet(path)
        first = pd.read_parquet(sorted(FT_DIR.glob("n9000_seed*.parquet"))[0])
        merged = first[["text"]].merge(llm[["text", "pred_intent"]], on="text", how="left")
        baseline_pred = merged.pred_intent.map(name_to_id).fillna(-1).astype(int).to_numpy()

    if baseline_pred is not None:
        print()
        print("=" * 78)
        print("3. Gap vs the LLM baseline: rows-only vs rows-and-seeds")
        print("=" * 78)
        print(f"  {'n':>6}{'gap':>10}{'rows-only 95% CI':>26}{'rows+seeds 95% CI':>26}")
        gaps = {}
        for n in SIZES:
            rows_only = hierarchical_bootstrap(
                y_true, ft_probs[n], baseline_pred, n_classes,
                n_bootstrap=args.bootstrap, resample_seeds=False)
            both = hierarchical_bootstrap(
                y_true, ft_probs[n], baseline_pred, n_classes,
                n_bootstrap=args.bootstrap, resample_seeds=True)
            perm = paired_permutation_test(
                y_true, ft_probs[n].mean(0).argmax(1), baseline_pred,
                n_classes, n_permutations=args.permutations, seed=42)
            gaps[n] = {"rows_only": rows_only, "rows_and_seeds": both,
                       "permutation_p_value": round(perm["p_value"], 5)}
            print(f"  {n:>6}{both['observed_diff']:>+10.4f}"
                  f"   [{rows_only['ci_lower_95']:+.4f}, {rows_only['ci_upper_95']:+.4f}]"
                  f"      [{both['ci_lower_95']:+.4f}, {both['ci_upper_95']:+.4f}]")
        summary["gap_vs_baseline"] = gaps

        print()
        print("=" * 78)
        print("4. Interval on the crossover budget itself")
        print("=" * 78)
        ci = crossover_interval(y_true, ft_probs, baseline_pred, n_classes,
                                n_bootstrap=args.bootstrap, resample_seeds=True)
        summary["crossover"] = ci
        if "median_crossover_n" in ci:
            print(f"  median n = {ci['median_crossover_n']}, "
                  f"95% CI [{ci['ci_lower_95_n']}, {ci['ci_upper_95_n']}]")
            print(f"  replicates never reaching parity: {ci['crossover_never_reached_fraction']:.1%}")
            print(f"  distribution across budgets: {ci['distribution']}")
        else:
            print(f"  {ci['note']}")
    else:
        print("\n  No --baseline supplied, so no LLM-relative statistics were computed.")
        print("  That is the correct default while the baseline is being rebuilt.")

    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
