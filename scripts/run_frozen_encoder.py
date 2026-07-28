"""Frozen-encoder baselines across every label budget (HANDOVER §6.2). No API key needed.

Why this arm exists, in the order the objections arrive:

  1. Fine-tuning DistilBERT from scratch is the wrong method for the low-budget
     regime. Without a frozen-feature baseline the encoder arm is a straw man at
     small n and the paper is trivially attackable.
  2. It is the cheapest available test of whether the crossover survives a
     competent small model at all -- a go/no-go probe for the 1 August gate,
     not merely one more row in a table.
  3. It costs nothing and needs no network once the encoder weights are cached,
     so it can run before the LLM arm is rebuilt.

Three encoders, chosen to bracket the space rather than to win:

  tfidf       char n-gram TF-IDF + logistic regression. The floor. If this is
              near the crossover, "you need a fine-tuned transformer" is in
              trouble and the paper needs to say so first.
  distilbert  mean-pooled distilbert-base-uncased, frozen. The *same* encoder as
              the fine-tuned arm, so the gap is exactly what fine-tuning bought
              -- no confound from swapping architectures.
  minilm      mean-pooled sentence-transformers/all-MiniLM-L6-v2, frozen. The
              arm §6.2 actually specifies and the one a reviewer expects. Loaded
              through `transformers`, so no new dependency.

Runs on any dataset in the harness registry. The budget/seed grid is discovered
from disk rather than hard-coded, because the sweeps differ: Banking77 is 8
budgets x 5 seeds, CLINC150 is 6 x 3. Same val set and test set as the
fine-tuned arm in both cases.

    python scripts/run_frozen_encoder.py --encoder tfidf distilbert minilm
    python scripts/run_frozen_encoder.py --dataset clinc150 --encoder minilm

Writes to results/frozen_encoder/<dataset>/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.harness import datasets  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results" / "frozen_encoder"

ENCODERS = {
    "tfidf": None,
    "distilbert": "distilbert-base-uncased",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}


def discover_subsets(data_dir: Path) -> dict[int, list[int]]:
    """Find the (budget -> seeds) grid actually present on disk.

    Hard-coding the grid would silently skip budgets on a dataset with a
    different sweep -- CLINC150 runs 6 budgets x 3 seeds, Banking77 8 x 5.
    """
    grid: dict[int, set[int]] = {}
    for path in data_dir.glob("train_n*_seed*.parquet"):
        stem = path.stem.replace("train_n", "")
        n_str, seed_str = stem.split("_seed")
        grid.setdefault(int(n_str), set()).add(int(seed_str))
    return {n: sorted(seeds) for n, seeds in sorted(grid.items())}


# --------------------------------------------------------------------------
# Featurisers
# --------------------------------------------------------------------------


def transformer_embeddings(texts: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    """Mean-pooled last hidden state over non-padding tokens, then L2-normalised.

    Attention-mask-weighted pooling, not a plain mean: padding tokens carry a
    hidden state and averaging over them would make the embedding depend on the
    batch's longest sequence.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model.to(device)

    out = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(device)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy())
    return np.vstack(out)


def build_features(encoder: str, corpus: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """Featurise every split at once. Fitting happens on the training pool only.

    Encodes each *distinct* text once. Every training subset is drawn from one pool,
    so the naive corpus repeats itself heavily -- on Banking77 that is ~104k rows over
    ~12.8k unique texts, an 8x waste that dominates transformer-encoder runtime.
    """
    # Sorted, not insertion-ordered. Transformer encoding runs in batches, and a
    # batch's padding length depends on which texts share it -- so insertion order
    # leaks into the embeddings through floating-point non-associativity. That is
    # not hypothetical: rebuilding the pool from a different set of files (same
    # texts, different order) moved one cell's regularisation choice and shifted a
    # published ensemble figure by 0.004. Sorting makes the embeddings a function
    # of the text set alone.
    unique = sorted({t for texts in corpus.values() for t in texts})
    index = {t: i for i, t in enumerate(unique)}
    take = {split: [index[t] for t in texts] for split, texts in corpus.items()}

    if encoder == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Fit on the training pool only -- never on test.
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
        vec.fit(corpus["pool"])
        matrix = vec.transform(unique)
    else:
        matrix = transformer_embeddings(unique, ENCODERS[encoder])

    print(f"  encoded {len(unique)} unique texts "
          f"({sum(len(v) for v in corpus.values())} total references)")
    return {split: matrix[rows] for split, rows in take.items()}


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="banking77", choices=sorted(datasets.REGISTRY))
    ap.add_argument("--encoder", nargs="+", default=["tfidf", "distilbert"],
                    choices=sorted(ENCODERS))
    ap.add_argument("--sizes", nargs="+", type=int, default=None,
                    help="Default: every budget found on disk for this dataset.")
    ap.add_argument("--C", type=float, default=None,
                    help="Fixed inverse regularisation. Default: tuned per cell on the val set.")
    ap.add_argument("--c-tol", type=float, default=0.0,
                    help="Prefer the smallest C within this macro-F1 margin of the best "
                         "validation score. 0 = strict argmax (with smaller C winning exact "
                         "ties). Offered for sensitivity analysis; it does NOT buy stability -- "
                         "determinism comes from sorting the encoded texts, not from this.")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    spec = datasets.get_dataset(args.dataset)
    data_dir = spec.data_dir
    names = list(spec.label_names)
    test, val = spec.load_split("test"), spec.load_split("val")

    grid = discover_subsets(data_dir)
    if args.sizes:
        grid = {n: seeds for n, seeds in grid.items() if n in args.sizes}
    if not grid:
        raise SystemExit(f"no training subsets found for {args.dataset} in {data_dir}")

    # Fit the vectoriser on every labelled training row available -- never on test.
    pool = pd.concat(
        [pd.read_parquet(data_dir / f"train_n{n}_seed{s}.parquet") for n, seeds in grid.items()
         for s in seeds]
    ).drop_duplicates(subset="text").reset_index(drop=True)

    out_dir = RESULTS / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.dataset}: {len(names)} classes, {len(test)} test rows, "
          f"{len(pool)} pooled training rows")
    print(f"grid: {({n: len(s) for n, s in grid.items()})}  (budget -> seeds)")

    all_rows: list[dict] = []
    ensembles: dict[tuple[str, int], float] = {}

    for encoder in args.encoder:
        print(f"\n=== {encoder} ===")
        started = time.perf_counter()
        corpus = {"pool": list(pool.text), "test": list(test.text), "val": list(val.text)}
        for n, seeds in grid.items():
            for seed in seeds:
                df = pd.read_parquet(data_dir / f"train_n{n}_seed{seed}.parquet")
                corpus[f"train_{n}_{seed}"] = list(df.text)
        feats = build_features(encoder, corpus)
        print(f"  featurised in {time.perf_counter() - started:.1f}s")

        y_test = test.label.to_numpy()
        y_val = val.label.to_numpy()

        for n, seeds in grid.items():
            scores, probs_for_n = [], []
            for seed in seeds:
                df = pd.read_parquet(data_dir / f"train_n{n}_seed{seed}.parquet")
                X, y = feats[f"train_{n}_{seed}"], df.label.to_numpy()

                # Regularisation is selected on the val set -- the same fixed
                # 500 rows the fine-tuned arm uses for early stopping, so both
                # arms are charged the same extra supervision (HANDOVER §6.4).
                if args.C is not None:
                    best_c, val_scores = args.C, {}
                else:
                    # Smallest C within `--c-tol` of the best validation score;
                    # with the default tol of 0 this is a strict argmax that gives
                    # exact ties to the more regularised model. Validation scores are
                    # recorded per cell so any near-tie is auditable after the fact.
                    val_scores = {}
                    for c in (0.1, 1.0, 10.0, 100.0):
                        probe = LogisticRegression(C=c, max_iter=2000).fit(X, y)
                        val_scores[c] = float(f1_score(
                            y_val, probe.predict(feats["val"]), average="macro",
                            labels=range(len(names)), zero_division=0))
                    ceiling = max(val_scores.values())
                    best_c = min(c for c, v in val_scores.items() if v >= ceiling - args.c_tol)

                clf = LogisticRegression(C=best_c, max_iter=2000).fit(X, y)
                proba = clf.predict_proba(feats["test"])
                # Classes absent from a small training subset are absent from the
                # model; re-expand to the full label simplex so ensembling and
                # macro-F1 are computed over the same space at every budget.
                full = np.zeros((len(y_test), len(names)))
                full[:, clf.classes_] = proba
                pred = full.argmax(1)
                macro = f1_score(y_test, pred, average="macro",
                                 labels=range(len(names)), zero_division=0)
                scores.append(macro)
                probs_for_n.append(full.astype(np.float32))
                all_rows.append({
                    "encoder": encoder, "training_size": n, "seed": seed,
                    "macro_f1": macro, "C": best_c,
                    "val_scores": json.dumps({str(k): round(v, 5) for k, v in val_scores.items()}),
                    "n_classes_seen": int(len(clf.classes_)),
                })
            # Soft-vote ensemble across seeds, matching the fine-tuned arm.
            stacked = np.stack(probs_for_n)          # (n_seeds, n_rows, n_classes)
            ens_probs = stacked.mean(0)
            ens = f1_score(y_test, ens_probs.argmax(1), average="macro",
                           labels=range(len(names)), zero_division=0)
            ensembles[(encoder, n)] = ens
            # Kept on disk rather than in the frame: n_rows x n_classes floats per
            # run is far too heavy to carry as a parquet column, and the
            # re-analysis needs the full simplex. Per-seed is saved too, because the
            # hierarchical bootstrap in src/stats.py resamples seeds and cannot
            # reconstruct them from an ensemble that has already been averaged.
            np.save(out_dir / f"probs_{encoder}_n{n}.npy", ens_probs)
            np.save(out_dir / f"probs_perseed_{encoder}_n{n}.npy", stacked)
            print(f"  n={n:>5}  seed-mean {np.mean(scores):.4f}  "
                  f"(sd {np.std(scores):.4f})  ensemble {ens:.4f}")

    frame = pd.DataFrame(all_rows)
    frame.to_parquet(out_dir / "frozen_predictions.parquet", index=False)

    summary = {}
    for encoder in args.encoder:
        sub = frame[frame.encoder == encoder]
        summary[encoder] = {
            "seed_mean_by_n": {
                int(n): round(float(g.macro_f1.mean()), 4) for n, g in sub.groupby("training_size")
            },
            "seed_sd_by_n": {
                int(n): round(float(g.macro_f1.std(ddof=0)), 4) for n, g in sub.groupby("training_size")
            },
            "ensemble_by_n": {
                int(n): round(float(ensembles[(encoder, n)]), 4)
                for n in sorted(sub.training_size.unique())
            },
        }
    (out_dir / "frozen_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
