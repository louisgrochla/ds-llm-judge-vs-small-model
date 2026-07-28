"""Regenerate the headline figure from committed artifacts, without the withdrawn LLM line.

`results/figures/crossover.png` has a red dashed baseline drawn at 0.8913 across it. That
number is invalid, and a figure is the part of a paper people read without reading the
caption — so a corrected figure is worth more than a corrected caption.

This produces `results/figures/label_budget_sweep.png`, which shows only what is measured:

  * fine-tuned DistilBERT, single-seed mean with the across-seed spread
  * fine-tuned DistilBERT, 5-seed soft-vote ensemble
  * every frozen-encoder arm that has been run
  * the x-axis as the *true* label budget (n + the 500-row validation set), because the
    validation set is supervision and the published axis excluded it

    python scripts/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stats import label_budget, macro_f1  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FT_DIR = ROOT / "results" / "finetune" / "v2_retune"   # banking77 only
FROZEN_ROOT = ROOT / "results" / "frozen_encoder"
FIG_DIR = ROOT / "results" / "figures"
SIZES = [50, 100, 250, 500, 1000, 2500, 5000, 9000]
CLINC_SIZES = [500, 1000, 2500, 5000, 9000, 13000]

STYLE = {
    "tfidf": ("char n-gram TF-IDF + logistic regression (frozen)", "#c2410c", "^", ":"),
    "distilbert": ("DistilBERT mean-pooled, frozen + logistic regression", "#0891b2", "v", ":"),
    "minilm": ("all-MiniLM-L6-v2 mean-pooled, frozen + logistic regression", "#7c3aed", "D", ":"),
}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="banking77", choices=("banking77", "clinc150"))
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.harness import datasets

    spec = datasets.get_dataset(args.dataset)
    n_classes = spec.n_classes
    sizes = SIZES if args.dataset == "banking77" else CLINC_SIZES
    frozen_dir = FROZEN_ROOT / args.dataset
    y_true = spec.load_split("test").label.to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6.2))

    # The fine-tuned arm exists for Banking77 only. CLINC150 is scoped as mechanism
    # replication (HANDOVER §6.3), and a fine-tuning sweep there would be a second
    # study. Plot what exists rather than implying a comparison that was not run.
    has_finetuned = args.dataset == "banking77"
    if has_finetuned:
        single_mean, single_sd, ensemble = [], [], []
        for n in sizes:
            frames = [pd.read_parquet(p) for p in sorted(FT_DIR.glob(f"n{n}_seed*.parquet"))]
            probs = np.stack([np.array(f.probs.tolist(), dtype=np.float32) for f in frames])
            per_seed = [macro_f1(y_true, probs[s].argmax(1), n_classes)
                        for s in range(len(frames))]
            single_mean.append(np.mean(per_seed))
            single_sd.append(np.std(per_seed))
            ensemble.append(macro_f1(y_true, probs.mean(0).argmax(1), n_classes))
        x = [label_budget(n) for n in sizes]
        single_mean, single_sd = np.array(single_mean), np.array(single_sd)
        ax.fill_between(x, single_mean - single_sd, single_mean + single_sd,
                        color="steelblue", alpha=0.18, label="fine-tuned, ±1 sd across 5 seeds")
        ax.plot(x, single_mean, color="steelblue", marker="o", linewidth=2,
                label="DistilBERT fine-tuned (single-seed mean)")
        ax.plot(x, ensemble, color="#1e3a8a", marker="s", linewidth=2.5,
                label="DistilBERT fine-tuned (5-seed soft-vote ensemble)")

    for enc, (label, colour, marker, dash) in STYLE.items():
        pts = [(label_budget(n), np.load(frozen_dir / f"probs_{enc}_n{n}.npy"))
               for n in sizes if (frozen_dir / f"probs_{enc}_n{n}.npy").exists()]
        if not pts:
            continue
        ax.plot([p[0] for p in pts],
                [macro_f1(y_true, p[1].argmax(1), n_classes) for p in pts],
                color=colour, marker=marker, linestyle=dash, linewidth=2, label=label)

    ticks = [label_budget(n) for n in sizes]
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    # Tick at the true budget and label it as such, with the nominal n underneath.
    # Labelling these positions with bare n is what the published figure did, and it
    # is what hid the 500 rows of validation supervision in the first place.
    ax.set_xticklabels([f"{label_budget(n):,}\n(n={n:,})" for n in sizes],
                       fontsize=8, rotation=45, ha="right")
    ax.set_xlabel("Labelled examples actually used  =  training subset n  +  the fixed 500-row "
                  "validation set", fontsize=11)
    pretty = "Banking77" if args.dataset == "banking77" else "CLINC150"
    n_rows = len(y_true)
    ax.set_ylabel(f"macro-F1 — {pretty} test, {n_rows:,} rows, {n_classes} classes", fontsize=11)
    ax.set_title(f"What each label budget buys on {pretty}", fontsize=14, pad=12)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    note = (
        "No LLM baseline is plotted: the one previously drawn here (0.8913) was collected\n"
        "with the test set ordered by class and has been withdrawn. Seeds at n≥5,000 share\n"
        "most of their training rows, so the shaded band narrows there for that reason."
        if has_finetuned else
        "Frozen encoders only. No fine-tuned arm was run on CLINC150 — this dataset is\n"
        "scoped as mechanism replication, not a second sweep — and no LLM baseline exists\n"
        "yet. Out-of-scope queries are held out of every figure on this dataset."
    )
    # Put the note wherever the curves are not. On Banking77 they start near zero so
    # the top-left is free; on CLINC150 every arm sits above 0.75 and the bottom-left is.
    lowest = min(line.get_ydata().min() for line in ax.get_lines() if len(line.get_ydata()))
    if lowest > 0.5:
        ax.text(0.015, 0.30, note, transform=ax.transAxes, fontsize=8, va="top",
                color="#4b5563", bbox=dict(boxstyle="round,pad=0.45", facecolor="#fffbeb",
                                           edgecolor="#d97706", lw=0.8))
    else:
        ax.text(0.015, 0.965, note, transform=ax.transAxes, fontsize=8, va="top",
                color="#4b5563", bbox=dict(boxstyle="round,pad=0.45", facecolor="#fffbeb",
                                           edgecolor="#d97706", lw=0.8))

    suffix = "" if args.dataset == "banking77" else f"_{args.dataset}"
    out = FIG_DIR / f"label_budget_sweep{suffix}.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
