"""Prepare CLINC150 as the second dataset (HANDOVER §6.3). Downloads once, no API key.

Scoped as **mechanism replication, not a second study**. The reframed claim is about
taxonomies, so it cannot rest on one taxonomy — but replicating the full 8-budget x
5-seed sweep would be a second study, and §9 puts that out of scope. What is needed is
the encoder at the largest budget with a few seeds, one LLM config, and the same
per-class decomposition.

CLINC150 is the right second dataset for three reasons: it sits with Banking77 in the
standard Casanueva trio, so reviewers read it as the standard evaluation rather than an
arbitrary addition; it is a *cleaner* taxonomy, which makes it a directional test rather
than a coin flip (the prediction is a **smaller** effect there, and a smaller effect
confirms that the effect scales with label-set confusability); and it has an explicit
out-of-scope split, which Banking77 lacks.

**Out-of-scope handling.** The `plus` config carries a 151st `oos` class with a different
sampling density from the 150 in-scope intents (250 train / 1,000 test vs 100 / 30).
Mixing it into a macro-F1 over 151 classes would let one anomalous class move the headline.
In-scope rows are written to the main splits; OOS rows are written separately to
`oos_test.parquet` for the out-of-scope analysis, which is a different question.

    python scripts/prepare_clinc150.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed_clinc150"
REPO = "clinc/clinc_oos"
CONFIG = "plus"
DEV_SLICE_SIZE = 300
VAL_SET_SIZE = 500
SIZES = [500, 1000, 2500, 5000, 9000, 13000]
N_SEEDS = 3


def load_raw() -> tuple[dict[str, pd.DataFrame], list[str]]:
    from datasets import load_dataset

    ds = load_dataset(REPO, CONFIG)
    names = list(ds["train"].features["intent"].names)
    frames = {
        split: pd.DataFrame({"text": ds[split]["text"], "intent_id": ds[split]["intent"]})
        for split in ("train", "validation", "test")
    }
    return frames, names


def stratified_take(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Take n rows, spread as evenly across intents as the pool allows."""
    rng = np.random.default_rng(seed)
    groups = [g.iloc[rng.permutation(len(g))] for _, g in df.groupby("label", sort=True)]
    picked, cursor = [], 0
    while sum(len(p) for p in picked) < n:
        added = False
        for g in groups:
            if cursor < len(g):
                picked.append(g.iloc[cursor : cursor + 1])
                added = True
                if sum(len(p) for p in picked) >= n:
                    break
        if not added:
            break
        cursor += 1
    return pd.concat(picked).iloc[:n].sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--seed", type=int, default=20260729)
    args = ap.parse_args()

    frames, all_names = load_raw()
    oos_candidates = [n for n in all_names if n.lower() in ("oos", "out_of_scope", "oos_intent")]
    if len(oos_candidates) != 1:
        raise RuntimeError(f"could not identify the out-of-scope class among {all_names[:5]}...")
    oos_name = oos_candidates[0]
    oos_id = all_names.index(oos_name)
    in_scope = [n for n in all_names if n != oos_name]
    remap = {all_names.index(n): i for i, n in enumerate(in_scope)}

    print(f"CLINC150 ({CONFIG}): {len(all_names)} classes, out-of-scope = {oos_name!r} (id {oos_id})")
    print(f"in-scope intents kept: {len(in_scope)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "label_names.txt").write_text("\n".join(in_scope) + "\n")

    def split_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        oos = df[df.intent_id == oos_id].reset_index(drop=True)
        keep = df[df.intent_id != oos_id].copy()
        keep["label"] = keep.intent_id.map(remap).astype(int)
        return keep[["text", "label"]].reset_index(drop=True), oos[["text"]].reset_index(drop=True)

    test, oos_test = split_frame(frames["test"])
    official_val, _ = split_frame(frames["validation"])
    train_pool, _ = split_frame(frames["train"])

    # CLINC150 ships a small number of texts that appear verbatim in both the train
    # and test splits. They are removed from the *training* side only: the official
    # test split is the comparable quantity and must not be edited. Leaving them in
    # would let a model score on rows it was trained on.
    dupes = sorted(set(train_pool.text) & set(test.text))
    if dupes:
        train_pool = train_pool[~train_pool.text.isin(dupes)].reset_index(drop=True)
        print(f"\nremoved {len(dupes)} text(s) duplicated between CLINC's train and test splits:")
        for d in dupes[:5]:
            print(f"  {d!r}")

    test.to_parquet(OUT_DIR / "test.parquet", index=False)
    oos_test.to_parquet(OUT_DIR / "oos_test.parquet", index=False)
    print(f"\ntest        {len(test):>6} rows, {test.label.nunique()} intents")
    print(f"oos_test    {len(oos_test):>6} rows (held out of every macro-F1)")

    # Carve val and dev from the *training* pool, mirroring Banking77, so the
    # official validation split stays untouched and available as a clean extra.
    shuffled = train_pool.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    val = stratified_take(shuffled, VAL_SET_SIZE, args.seed)
    remaining = shuffled[~shuffled.text.isin(set(val.text))].reset_index(drop=True)
    dev = stratified_take(remaining, DEV_SLICE_SIZE, args.seed + 1)
    pool = remaining[~remaining.text.isin(set(dev.text))].reset_index(drop=True)

    val.to_parquet(OUT_DIR / "val.parquet", index=False)
    dev.to_parquet(OUT_DIR / "dev_slice.parquet", index=False)
    pool.to_parquet(OUT_DIR / "demo_pool.parquet", index=False)
    official_val.to_parquet(OUT_DIR / "official_val.parquet", index=False)
    print(f"val         {len(val):>6} rows (carved from train, for early stopping)")
    print(f"dev_slice   {len(dev):>6} rows (carved from train, for the prompt ladder)")
    print(f"demo_pool   {len(pool):>6} rows (k-shot source and training pool)")
    print(f"official_val{len(official_val):>6} rows (untouched CLINC validation split)")

    print(f"\ntraining subsets ({args.seeds} seeds each):")
    written = []
    for n in args.sizes:
        if n > len(pool):
            print(f"  n={n:>6}  SKIPPED — exceeds the {len(pool)}-row pool")
            continue
        for seed in range(args.seeds):
            sub = stratified_take(pool, n, seed)
            sub.to_parquet(OUT_DIR / f"train_n{n}_seed{seed}.parquet", index=False)
        written.append(n)
        print(f"  n={n:>6}  written")

    overlaps = {
        "dev_vs_test": len(set(dev.text) & set(test.text)),
        "val_vs_test": len(set(val.text) & set(test.text)),
        "pool_vs_test": len(set(pool.text) & set(test.text)),
        "dev_vs_pool": len(set(dev.text) & set(pool.text)),
        "val_vs_pool": len(set(val.text) & set(pool.text)),
    }
    print("\nsplit disjointness (all must be 0):")
    for k, v in overlaps.items():
        print(f"  {'ok  ' if v == 0 else 'LEAK'}  {k:<16} {v}")

    (OUT_DIR / "manifest.json").write_text(json.dumps({
        "source": f"{REPO}/{CONFIG}",
        "n_classes_in_scope": len(in_scope),
        "out_of_scope_class": oos_name,
        "out_of_scope_rows_held_out": {"test": int(len(oos_test))},
        "splits": {"test": len(test), "val": len(val), "dev_slice": len(dev),
                   "demo_pool": len(pool), "official_val": len(official_val)},
        "training_sizes": written,
        "n_seeds": args.seeds,
        "carve_seed": args.seed,
        "overlaps": overlaps,
        "scope": "mechanism replication only (HANDOVER §6.3) — not a full sweep",
    }, indent=2) + "\n")

    if any(overlaps.values()):
        print("\nFAILED: splits overlap")
        return 1
    print(f"\nwrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
