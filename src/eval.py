"""Evaluation helpers shared across notebooks 02, 03, 04."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def compute_metrics(y_true, y_pred, labels=None) -> dict:
    """Return accuracy, macro-F1, weighted-F1, and per-class F1 in one dict."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        "per_class_f1": f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist(),
    }


def paired_bootstrap_macro_f1(
    y_true,
    y_pred_a,
    y_pred_b,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap on macro-F1 difference (A minus B), resampling test instances.

    Use this to test whether DistilBERT (A) actually matches/beats the LLM (B)
    at the candidate crossover n. A 95% CI that crosses zero means the gap is
    not statistically significant.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    n = len(y_true)

    observed = (
        f1_score(y_true, y_pred_a, average="macro", zero_division=0)
        - f1_score(y_true, y_pred_b, average="macro", zero_division=0)
    )

    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[i] = (
            f1_score(y_true[idx], y_pred_a[idx], average="macro", zero_division=0)
            - f1_score(y_true[idx], y_pred_b[idx], average="macro", zero_division=0)
        )

    return {
        "observed_diff": float(observed),
        "bootstrap_mean": float(diffs.mean()),
        "ci_lower_95": float(np.quantile(diffs, 0.025)),
        "ci_upper_95": float(np.quantile(diffs, 0.975)),
        "n_bootstrap": n_bootstrap,
    }


def load_llm_predictions() -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / "llm_baseline_predictions.parquet")


def load_finetune_predictions() -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / "finetune_predictions.parquet")
