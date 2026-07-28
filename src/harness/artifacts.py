"""Run manifests, artifact writing, and cost computed from measured tokens.

Every run writes results/llm_runs/<run_id>/ containing predictions.parquet and
manifest.json. The manifest pins everything needed to reproduce or dispute a
number: model requested *and* the model id the API actually served, prompt
fingerprint, seeds, ordering, demonstration count, SDK version, git sha.

Cost is derived from recorded `usage`, never estimated from row counts. The
price table is a dated snapshot and is treated as one -- `PRICING_RETRIEVED_ON`
travels with every artifact so a stale rate can never be quoted as current.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "llm_runs"

# USD per million tokens, standard (non-batch) synchronous rates.
# Source: Anthropic pricing as documented on the date below. RE-VERIFY before the
# paper: HANDOVER §8 requires one cost figure with one stated utilisation
# assumption and the date pricing was retrieved.
PRICING_RETRIEVED_ON = "2026-06-24"
PRICING = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25
BATCH_MULTIPLIER = 0.5


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


@dataclass
class RunManifest:
    run_id: str
    created_at: str
    dataset: str
    split: str
    n_rows: int
    model: str
    execution: str
    rung: str
    k: int
    queries_per_call: int
    order: str
    shuffle_seed: int
    runs: int
    thinking: str
    effort: str | None
    n_demonstrations: int
    prompt_fingerprint: str
    n_classes: int
    anthropic_sdk_version: str
    git_sha: str | None = field(default_factory=git_sha)
    pricing_retrieved_on: str = PRICING_RETRIEVED_ON
    resolved_models: list[str] = field(default_factory=list)
    batch_ids: list[str] = field(default_factory=list)
    notes: str = ""

    def write(self, directory: Path) -> Path:
        path = directory / "manifest.json"
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
        return path


def cost_usd(predictions: pd.DataFrame, model: str, execution: str) -> dict:
    """Cost from measured tokens. Per-call fields are de-duplicated first.

    Usage is attached to every prediction row, but a batched call reports its
    usage once -- summing the rows would multiply it by queries_per_call.
    """
    if model not in PRICING:
        return {"error": f"no price on file for {model!r} as of {PRICING_RETRIEVED_ON}"}
    per_call = predictions.drop_duplicates(subset=["run_index", "call_id"])
    rate = PRICING[model]
    inp = per_call.input_tokens.fillna(0).sum()
    out = per_call.output_tokens.fillna(0).sum()
    cache_read = per_call.cache_read_input_tokens.fillna(0).sum()
    cache_write = per_call.cache_creation_input_tokens.fillna(0).sum()
    multiplier = BATCH_MULTIPLIER if execution == "batch" else 1.0
    total = multiplier * (
        inp / 1e6 * rate["input"]
        + out / 1e6 * rate["output"]
        + cache_read / 1e6 * rate["input"] * CACHE_READ_MULTIPLIER
        + cache_write / 1e6 * rate["input"] * CACHE_WRITE_MULTIPLIER
    )
    n_queries = len(predictions)
    return {
        "model": model,
        "execution": execution,
        "pricing_retrieved_on": PRICING_RETRIEVED_ON,
        "batch_discount_applied": execution == "batch",
        "input_tokens": int(inp),
        "output_tokens": int(out),
        "cache_read_input_tokens": int(cache_read),
        "cache_creation_input_tokens": int(cache_write),
        "total_usd": round(float(total), 4),
        "usd_per_1k_queries": round(float(total) / max(n_queries, 1) * 1000, 4),
    }


def latency_summary(predictions: pd.DataFrame) -> dict | None:
    """p50/p95 per *call*, not per row. Mean is deliberately not the headline."""
    timed = predictions.dropna(subset=["latency_ms"]).drop_duplicates(
        subset=["run_index", "call_id"]
    )
    if timed.empty:
        return None
    ms = timed.latency_ms.to_numpy()
    return {
        "n_calls": int(len(ms)),
        "p50_ms": round(float(np.percentile(ms, 50)), 1),
        "p95_ms": round(float(np.percentile(ms, 95)), 1),
        "p99_ms": round(float(np.percentile(ms, 99)), 1),
        "min_ms": round(float(ms.min()), 1),
        "max_ms": round(float(ms.max()), 1),
    }


def run_variance(predictions: pd.DataFrame) -> dict:
    """How often repeated runs of the same row disagree. Determinism is not assumed."""
    runs = sorted(predictions.run_index.unique())
    if len(runs) < 2:
        return {"n_runs": len(runs), "note": "needs >= 2 runs to measure disagreement"}
    wide = predictions.pivot_table(
        index="row_id", columns="run_index", values="pred_intent", aggfunc="first"
    )
    complete = wide.dropna()
    unanimous = complete.nunique(axis=1).eq(1).mean() if len(complete) else float("nan")
    return {
        "n_runs": len(runs),
        "n_rows_with_all_runs": int(len(complete)),
        "unanimous_fraction": round(float(unanimous), 4),
        "disagreement_fraction": round(1 - float(unanimous), 4),
    }


def write_run(
    run_id: str, manifest: RunManifest, predictions: pd.DataFrame, summary: dict
) -> Path:
    directory = RESULTS_DIR / run_id
    directory.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(directory / "predictions.parquet", index=False)
    manifest.write(directory)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return directory
