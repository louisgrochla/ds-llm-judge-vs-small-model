"""Run the LLM arm of the study (HANDOVER §6.1). Dataset-agnostic entrypoint.

    # inspect what would be sent, without touching the API or spending anything
    python scripts/run_llm_eval.py --dataset banking77 --split dev --dry-run

    # a rung of the prompt ladder on the 300-row dev slice
    python scripts/run_llm_eval.py --split dev --rung glosses --runs 1

    # the headline quality run: per-query, shuffled, 3 runs, via the Batch API
    python scripts/run_llm_eval.py --split test --rung <frozen> --order shuffled --runs 3

    # the ablation: same rows, same prompt, 150 queries per call, both orders
    python scripts/run_llm_eval.py --split test --queries-per-call 150 --order blocked
    python scripts/run_llm_eval.py --split test --queries-per-call 150 --order shuffled

    # latency, which the Batch API cannot measure: sequential, timed, subsample
    python scripts/run_llm_eval.py --split test --execution sync --limit 200 --runs 1

Writes results/llm_runs/<run_id>/{predictions.parquet,manifest.json,summary.json}.
Nothing here scores the crossover -- that is re-analysis (§6.4) against these artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.harness import artifacts, datasets, prompting, runner  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="banking77", choices=sorted(datasets.REGISTRY))
    p.add_argument("--split", default="dev")
    p.add_argument("--model", default="claude-sonnet-4-6",
                   help="Pinned model id, recorded in the manifest alongside the id the API served.")
    p.add_argument("--rung", default="zero_shot", choices=prompting.RUNGS)
    p.add_argument("-k", type=int, default=0, help="Demonstrations per class (fixed) or per query (retrieved).")
    p.add_argument("--queries-per-call", type=int, default=1,
                   help="1 = the deployment the cost model prices. >1 reproduces the v1 batched collection.")
    p.add_argument("--order", default="shuffled", choices=runner.ORDERS)
    p.add_argument("--runs", type=int, default=1, help="Repeat identical calls to measure run variance.")
    p.add_argument("--execution", default="batch", choices=("batch", "sync"))
    p.add_argument("--thinking", default="disabled", choices=("disabled", "adaptive"))
    p.add_argument("--effort", default=None, choices=("low", "medium", "high", "xhigh", "max"))
    p.add_argument("--shuffle-seed", type=int, default=20260729)
    p.add_argument("--demo-seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows after ordering.")
    p.add_argument("--run-id", default=None)
    p.add_argument("--notes", default="")
    p.add_argument("--dry-run", action="store_true",
                   help="Assemble every call and report, without calling the API.")
    return p.parse_args(argv)


def score(predictions: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """Macro-F1 per run. Unparseable/missing predictions count as wrong, and are counted."""
    from sklearn.metrics import f1_score

    truth = gold.set_index("row_id").intent
    out = {}
    for run_index, group in predictions.groupby("run_index"):
        y_true = truth.loc[group.row_id].to_numpy()
        y_pred = group.pred_intent.fillna("__missing__").to_numpy()
        out[int(run_index)] = {
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "accuracy": round(float((y_true == y_pred).mean()), 4),
            "n_missing": int(group.pred_intent.isna().sum()),
            "n_rows": int(len(group)),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = datasets.get_dataset(args.dataset)
    cfg = prompting.PromptConfig(
        rung=args.rung,
        k=args.k,
        queries_per_call=args.queries_per_call,
        thinking=args.thinking,
        effort=args.effort,
    )

    rows = spec.load_split(args.split)
    glosses = spec.load_glosses() if cfg.uses_glosses else None

    fixed_demos = retriever = None
    n_demonstrations = 0
    if cfg.rung in ("fixed_kshot", "retrieved_kshot"):
        pool = spec.load_split(spec.demo_pool_split)
        if cfg.rung == "fixed_kshot":
            fixed_demos = prompting.fixed_demos(pool, spec.label_names, cfg.k, args.demo_seed)
            n_demonstrations = len(fixed_demos)
        else:
            retriever = prompting.TfidfRetriever(pool, cfg.k)
            n_demonstrations = cfg.k  # per query, drawn from a labelled pool of len(pool)

    calls = runner.assemble_calls(
        spec.label_names, rows, cfg, args.order, args.shuffle_seed, glosses, fixed_demos, retriever
    )
    if args.limit is not None:
        keep = -(-args.limit // cfg.queries_per_call)
        calls = calls[:keep]
    covered = [rid for call in calls for rid in call.row_ids]
    fingerprint = prompting.prompt_fingerprint(calls[0].body, cfg)

    run_id = args.run_id or "-".join(
        [args.dataset, args.split, args.model, cfg.rung, f"q{cfg.queries_per_call}",
         args.order, f"x{args.runs}", args.execution, fingerprint]
    )

    print(f"dataset       {spec.name} / {args.split}  ({len(rows)} rows, {spec.n_classes} classes)")
    print(f"evaluating    {len(covered)} rows in {len(calls)} calls x {args.runs} run(s)")
    print(f"prompt        {cfg.rung} k={cfg.k} thinking={cfg.thinking} effort={cfg.effort}")
    print(f"order         {args.order} (seed {args.shuffle_seed})")
    print(f"model         {args.model} via {args.execution}")
    print(f"fingerprint   {fingerprint}")
    print(f"demos         {n_demonstrations} labelled example(s) -- counts toward the label budget")

    if args.dry_run:
        out = artifacts.RESULTS_DIR / run_id
        out.mkdir(parents=True, exist_ok=True)
        preview = {
            "run_id": run_id,
            "n_calls": len(calls),
            "n_rows_covered": len(covered),
            "max_tokens_per_call": prompting.max_tokens_for(cfg.queries_per_call),
            "system_chars": len(calls[0].body["system"][0]["text"]),
            "first_call": calls[0].body,
            "last_call_row_ids": calls[-1].row_ids,
        }
        (out / "dry_run.json").write_text(json.dumps(preview, indent=2) + "\n")
        print(f"\nDRY RUN -- no API calls made. Assembled request written to {out / 'dry_run.json'}")
        return 0

    import anthropic

    client = anthropic.Anthropic()
    all_rows: list[dict] = []
    batch_ids: list[str] = []
    for run_index in range(args.runs):
        print(f"\nrun {run_index + 1}/{args.runs}")
        if args.execution == "batch":
            produced, batch_id = runner.run_batch(
                client, calls, args.model, cfg, progress=lambda m: print(f"  {m}")
            )
            batch_ids.append(batch_id)
        else:
            produced = runner.run_sync(
                client, calls, args.model, cfg,
                progress=lambda i, n: print(f"  {i}/{n}", end="\r", flush=True),
            )
        all_rows += [{**row, "run_index": run_index} for row in produced]

    predictions = pd.DataFrame(all_rows).merge(
        rows[["row_id", "text", "label", "intent"]].rename(columns={"intent": "true_intent"}),
        on="row_id",
        how="left",
    )
    predictions["dataset"] = spec.name
    predictions["split"] = args.split
    predictions["order"] = args.order
    predictions["queries_per_call"] = cfg.queries_per_call
    predictions["rung"] = cfg.rung

    summary = {
        "per_run": score(predictions, rows),
        "run_variance": artifacts.run_variance(predictions),
        "cost": artifacts.cost_usd(predictions, args.model, args.execution),
        "latency": artifacts.latency_summary(predictions),
    }
    manifest = artifacts.RunManifest(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset=spec.name,
        split=args.split,
        n_rows=len(covered),
        model=args.model,
        execution=args.execution,
        rung=cfg.rung,
        k=cfg.k,
        queries_per_call=cfg.queries_per_call,
        order=args.order,
        shuffle_seed=args.shuffle_seed,
        runs=args.runs,
        thinking=cfg.thinking,
        effort=cfg.effort,
        n_demonstrations=n_demonstrations,
        prompt_fingerprint=fingerprint,
        n_classes=spec.n_classes,
        anthropic_sdk_version=anthropic.__version__,
        resolved_models=sorted(predictions.resolved_model.dropna().unique().tolist()),
        batch_ids=batch_ids,
        notes=args.notes,
    )
    directory = artifacts.write_run(run_id, manifest, predictions, summary)
    print(f"\nwrote {directory}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
