"""Re-derive every load-bearing number in HANDOVER.md §2 from the committed artifacts.

HANDOVER.md §13 rule 2: re-derive before restating. Multiple independent reviews of
this project produced conflicting figures for the same quantities; computation settles
it, consensus does not. Run this before quoting any §2 number in the paper.

    python scripts/verify_handover_claims.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LLM_PRED = "results/llm_predictions_v1_test.parquet"
FT_PRED = "results/finetune_predictions_v2_retune.parquet"

# The five near-synonymous card/top-up intents where Banking77's labelling
# convention is arbitrary. Selected as the top-5 by leave-one-out contribution
# to the encoder's lead -- the jackknife below reports that selection honestly.
CONVENTION_CLASSES = [
    "get_physical_card",
    "order_physical_card",
    "beneficiary_not_allowed",
    "topping_up_by_card",
    "top_up_by_bank_transfer_charge",
]


def macro_f1(y_true, y_pred, labels=None):
    return f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)


def ensemble_predictions(ft, n, class_names):
    """Soft-vote across seeds: mean softmax, then argmax."""
    g = ft[ft.training_size == n]
    probs = np.stack(
        [np.stack(gs.sort_values("text").probs.values) for _, gs in g.groupby("seed")]
    ).mean(0)
    base = g[g.seed == 0].sort_values("text")
    return base.true_intent.values, np.array([class_names[i] for i in probs.argmax(1)])


def main():
    llm = pd.read_parquet(LLM_PRED)
    ft = pd.read_parquet(FT_PRED)
    class_names = sorted(
        ft.true_intent.unique(),
        key=lambda c: ft.loc[ft.true_intent == c, "true_label"].iloc[0],
    )
    kept = [c for c in class_names if c not in CONVENTION_CLASSES]

    llm_sorted = llm.sort_values("text")
    lt, lp = llm_sorted.true_intent.values, llm_sorted.pred_intent.values
    llm_all = macro_f1(lt, lp)

    # -- 2.1 -------------------------------------------------------------
    print("=" * 72)
    print("2.1  LLM baseline validity: was the test set fed in sorted by class?")
    print("=" * 72)
    ordered = llm.sort_values(["batch", "pred_id"]).true_intent.values
    transitions = sum(ordered[i] != ordered[i - 1] for i in range(1, len(ordered)))
    expected_random = int(len(ordered) * (1 - 1 / len(class_names)))
    per_batch = llm.groupby("batch").agg(
        rows=("text", "size"),
        intents=("true_intent", "nunique"),
        accuracy=("is_correct", "mean"),
    )
    print(f"  rows                        {len(ordered)}")
    print(f"  label transitions           {transitions}   (random order ~{expected_random})")
    print(f"  distinct classes            {len(set(ordered))}")
    print(f"  mean intents per batch      {per_batch.intents.mean():.2f}")
    print(f"  per-batch accuracy range    {per_batch.accuracy.min():.3f} - {per_batch.accuracy.max():.3f}")
    verdict = "SORTED BY CLASS -- baseline invalid" if transitions < 200 else "shuffled"
    print(f"  VERDICT                     {verdict}")

    # -- 2.2 -------------------------------------------------------------
    print()
    print("=" * 72)
    print(f"2.2  Per-seed crossover vs LLM macro-F1 {llm_all:.4f} (all 77 classes)")
    print("=" * 72)
    header = f"{'n':>6}" + "".join(f"{'s' + str(s):>8}" for s in range(5))
    print(header + f"{'mean':>8}{'ens':>8}{'beat':>6}")
    for n in sorted(ft.training_size.unique()):
        g = ft[ft.training_size == n]
        per_seed = [macro_f1(gs.true_intent, gs.pred_intent) for _, gs in g.groupby("seed")]
        yt, yp = ensemble_predictions(ft, n, class_names)
        n_beat = sum(p > llm_all for p in per_seed)
        flag = "  <-- ensemble-only crossover" if n_beat == 0 and macro_f1(yt, yp) > llm_all else ""
        print(
            f"{n:>6}" + "".join(f"{p:>8.4f}" for p in per_seed)
            + f"{np.mean(per_seed):>8.4f}{macro_f1(yt, yp):>8.4f}{n_beat:>6}{flag}"
        )

    # -- 2.3 -------------------------------------------------------------
    print()
    print("=" * 72)
    print("2.3  Sensitivity to the five annotation-convention classes")
    print("=" * 72)
    print("  A = mean F1 over the 72 kept classes, all 3080 rows retained")
    print("  B = A, but also dropping rows whose TRUE label is excluded")
    llm_mask = ~pd.Series(lt).isin(CONVENTION_CLASSES).values
    llm_a = macro_f1(lt, lp, kept)
    llm_b = macro_f1(lt[llm_mask], lp[llm_mask], kept)
    print(f"\n  LLM: all-77 {llm_all:.4f} | A {llm_a:.4f} | B {llm_b:.4f}\n")
    print(f"{'n':>6}{'gap(77)':>10}{'ensA':>9}{'gapA':>10}{'ensB':>9}{'gapB':>10}")
    for n in sorted(ft.training_size.unique()):
        yt, yp = ensemble_predictions(ft, n, class_names)
        mask = ~pd.Series(yt).isin(CONVENTION_CLASSES).values
        a, b = macro_f1(yt, yp, kept), macro_f1(yt[mask], yp[mask], kept)
        print(
            f"{n:>6}{macro_f1(yt, yp) - llm_all:>+10.4f}"
            f"{a:>9.4f}{a - llm_a:>+10.4f}{b:>9.4f}{b - llm_b:>+10.4f}"
        )

    # -- fairness check ---------------------------------------------------
    print()
    print("=" * 72)
    print("Fairness check: leave-one-class-out jackknife of the n=9000 gap")
    print("=" * 72)
    yt, yp = ensemble_predictions(ft, 9000, class_names)
    full_gap = macro_f1(yt, yp) - llm_all
    jack = pd.Series(
        {
            c: macro_f1(yt, yp, [x for x in class_names if x != c])
            - macro_f1(lt, lp, [x for x in class_names if x != c])
            for c in class_names
        }
    ).sort_values()
    print(f"  full-77 gap {full_gap:+.4f}   jackknife range {jack.min():+.4f} .. {jack.max():+.4f}")
    print("  five classes whose removal shrinks the encoder's lead most:")
    for c, v in jack.head(5).items():
        marker = "*" if c in CONVENTION_CLASSES else " "
        print(f"   {marker} -{c:<38}{v:+.4f}")
    print("    (* = member of CONVENTION_CLASSES; the exclusion set is selected on")
    print("     outcome, so report it as a sensitivity analysis, not a correction.)")
    share = 1 - (macro_f1(yt, yp, kept) - llm_a) / full_gap
    print(f"\n  share of the n=9000 lead attributable to those 5 of 77 classes: {share:.0%}")


if __name__ == "__main__":
    main()
