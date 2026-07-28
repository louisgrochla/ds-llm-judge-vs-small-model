"""Inference for the re-analysis (HANDOVER §6.4). Three things `src/eval.py` cannot do.

1. **An actual p-value.** The repo published "*p* < 0.01 paired bootstrap" while computing
   only confidence intervals. A CI excluding zero is not a p-value. `paired_permutation_test`
   implements approximate randomisation (Yeh 2000; Dror et al. 2018) -- the standard paired
   test for two systems on the same test set, and one that makes no distributional assumption
   about macro-F1, which is neither a mean nor asymptotically normal.

2. **Uncertainty over seeds as well as rows.** The published bootstrap resampled test rows
   only, holding the 5 training seeds fixed. That treats "the ensemble" as a fixed system and
   understates uncertainty wherever seed choice matters -- which, per
   `scripts/check_split_integrity.py`, is exactly the low-budget regime. `hierarchical_bootstrap`
   resamples seeds *and* rows.

3. **An interval on the crossover point itself.** The claim of interest is "parity at n = X",
   so X is the estimate that needs an interval. Per-budget CIs do not give you one.
   `crossover_interval` reports the distribution of X across bootstrap replicates.

All three take predicted *label ids*, so they work identically for the encoder arms, the frozen
arms, and the rebuilt LLM arm.
"""

from __future__ import annotations

import numpy as np


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-F1 over a fixed label space, matching sklearn's `labels=range(n_classes),
    zero_division=0`. Bincount-based so it is fast enough for 10k permutations."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    correct = y_true == y_pred
    tp = np.bincount(y_true[correct], minlength=n_classes)[:n_classes]
    pred_counts = np.bincount(y_pred, minlength=n_classes)[:n_classes]
    true_counts = np.bincount(y_true, minlength=n_classes)[:n_classes]
    denom = 2 * tp + (pred_counts - tp) + (true_counts - tp)
    return float(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0).mean())


def paired_permutation_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_classes: int,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> dict:
    """Two-sided approximate randomisation test on the macro-F1 difference (A − B).

    Under the null the two systems are interchangeable, so for each test row we swap their
    predictions with probability 0.5 and recompute the difference. The p-value is the share
    of permutations whose |difference| is at least the observed one, with the standard +1
    correction that keeps p strictly positive (a p of exactly 0 is not a claim the test can
    support at any finite number of permutations).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    a = np.asarray(y_pred_a)
    b = np.asarray(y_pred_b)
    if not (len(y_true) == len(a) == len(b)):
        raise ValueError("y_true, y_pred_a and y_pred_b must be the same length")

    observed = macro_f1(y_true, a, n_classes) - macro_f1(y_true, b, n_classes)
    at_least_as_extreme = 0
    for _ in range(n_permutations):
        swap = rng.random(len(y_true)) < 0.5
        perm_a = np.where(swap, b, a)
        perm_b = np.where(swap, a, b)
        diff = macro_f1(y_true, perm_a, n_classes) - macro_f1(y_true, perm_b, n_classes)
        if abs(diff) >= abs(observed):
            at_least_as_extreme += 1

    return {
        "observed_diff": observed,
        "p_value": (at_least_as_extreme + 1) / (n_permutations + 1),
        "n_permutations": n_permutations,
        "test": "two-sided paired approximate randomisation on macro-F1",
    }


def hierarchical_bootstrap(
    y_true: np.ndarray,
    seed_probs: np.ndarray,
    baseline_pred: np.ndarray,
    n_classes: int,
    n_bootstrap: int = 2000,
    seed: int = 42,
    resample_seeds: bool = True,
) -> dict:
    """Bootstrap the ensemble-minus-baseline macro-F1 gap over seeds and test rows.

    `seed_probs` is (n_seeds, n_rows, n_classes). Each replicate draws n_seeds seeds with
    replacement, soft-votes them, then draws n_rows test rows with replacement. Setting
    `resample_seeds=False` reproduces the rows-only interval the repo published, so the two
    can be compared directly -- the difference between them is the uncertainty the original
    analysis omitted.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    baseline_pred = np.asarray(baseline_pred)
    n_seeds, n_rows, _ = seed_probs.shape

    observed = (
        macro_f1(y_true, seed_probs.mean(0).argmax(1), n_classes)
        - macro_f1(y_true, baseline_pred, n_classes)
    )

    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        chosen = rng.integers(0, n_seeds, n_seeds) if resample_seeds else np.arange(n_seeds)
        ens = seed_probs[chosen].mean(0).argmax(1)
        rows = rng.integers(0, n_rows, n_rows)
        diffs[i] = (
            macro_f1(y_true[rows], ens[rows], n_classes)
            - macro_f1(y_true[rows], baseline_pred[rows], n_classes)
        )

    return {
        "observed_diff": float(observed),
        "ci_lower_95": float(np.quantile(diffs, 0.025)),
        "ci_upper_95": float(np.quantile(diffs, 0.975)),
        "n_bootstrap": n_bootstrap,
        "resampled_seeds": resample_seeds,
    }


def crossover_interval(
    y_true: np.ndarray,
    probs_by_n: dict[int, np.ndarray],
    baseline_pred: np.ndarray,
    n_classes: int,
    n_bootstrap: int = 2000,
    seed: int = 42,
    resample_seeds: bool = True,
) -> dict:
    """Interval on the crossover budget itself: the smallest n at which the system reaches parity.

    `probs_by_n` maps budget -> (n_seeds, n_rows, n_classes). Each replicate resamples seeds
    and rows *once* and reuses that draw across every budget, so the curve stays internally
    consistent -- resampling independently per budget would let the crossover jump around for
    reasons that have nothing to do with the budget.

    Replicates where no budget reaches parity are reported separately rather than dropped: if
    that happens often, "the crossover is at n = X" is not a well-defined claim, and the honest
    output is the failure rate, not a tighter-looking interval computed on the survivors.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    baseline_pred = np.asarray(baseline_pred)
    budgets = sorted(probs_by_n)
    n_seeds, n_rows, _ = probs_by_n[budgets[0]].shape

    crossovers: list[int] = []
    never = 0
    for _ in range(n_bootstrap):
        chosen = rng.integers(0, n_seeds, n_seeds) if resample_seeds else np.arange(n_seeds)
        rows = rng.integers(0, n_rows, n_rows)
        base = macro_f1(y_true[rows], baseline_pred[rows], n_classes)
        found = None
        for n in budgets:
            ens = probs_by_n[n][chosen].mean(0).argmax(1)
            if macro_f1(y_true[rows], ens[rows], n_classes) >= base:
                found = n
                break
        if found is None:
            never += 1
        else:
            crossovers.append(found)

    if not crossovers:
        return {
            "crossover_never_reached_fraction": 1.0,
            "note": "parity was not reached at any budget in any replicate",
            "n_bootstrap": n_bootstrap,
        }
    arr = np.array(crossovers)
    return {
        "median_crossover_n": int(np.median(arr)),
        "ci_lower_95_n": int(np.quantile(arr, 0.025)),
        "ci_upper_95_n": int(np.quantile(arr, 0.975)),
        "crossover_never_reached_fraction": round(never / n_bootstrap, 4),
        "distribution": {int(n): int((arr == n).sum()) for n in budgets},
        "n_bootstrap": n_bootstrap,
        "resampled_seeds": resample_seeds,
    }


def label_budget(n: int, val_set_size: int = 500) -> int:
    """Total labelled examples a budget actually consumes.

    The fixed validation set used for early stopping at every budget is supervision, and the
    published x-axis did not count it. At n=50 that is a 11x understatement of the true cost.
    """
    return n + val_set_size
