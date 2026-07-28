"""Serving cost and latency for every in-house arm, measured rather than assumed.

The published cost model rested on two numbers with no artifact behind them --
"200 inferences/second at batch=8 on a T4" and a 10% utilisation figure -- and on
a latency claim ("~30ms vs ~1.5s, ~50x faster") that was never measured at all.
This script measures what can be measured on the machine it runs on, and is
explicit about what remains an assumption.

What it measures, end to end, per query, as a deployment would serve it:

  tfidf        vectorise + logistic regression        (CPU only, no GPU needed)
  minilm       embed + logistic regression            (frozen, 22M params)
  distilbert   embed + logistic regression            (frozen, 66M params)
  finetuned    the fine-tuned classifier itself       (66M params)

Both batch=1 (the interactive path, which is what latency means to a user) and
a throughput path (batched, which is what cost depends on).

What it cannot measure here: the API arm. That needs a key, and the harness's
synchronous path (`run_llm_eval.py --execution sync`) reports its p50/p95 with
`usage` attached, so the API side becomes measured too rather than estimated.

    python scripts/cost_model.py
    python scripts/cost_model.py --device cpu --repeats 5
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
CKPT = ROOT / "results" / "checkpoints" / "v2_retune_n9000_seed0"
OUT = ROOT / "results" / "cost_model.json"

# Instance prices, USD/hour. List prices -- these are quotes, not measurements,
# and they are the input most likely to be stale. Date them when you change them.
PRICES_RETRIEVED = "2026-05-24"
GPU_HOURLY = 0.50      # HF Inference Endpoint, T4
CPU_HOURLY = 0.05      # a small always-on CPU instance, order of magnitude
SECONDS_PER_MONTH = 24 * 30 * 3600

# The API arm, still estimated. Replaced by measured `usage` once the key lands.
API_USD_PER_QUERY_UNCACHED = 0.00525


def _load_average() -> float | None:
    try:
        import os

        return os.getloadavg()[0]
    except (OSError, AttributeError):
        return None


def timed(fn, repeats: int) -> dict:
    """Run fn `repeats` times, discarding a warmup, and report the distribution."""
    fn()  # warmup: first call pays import, lazy-init and allocation costs
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "p50_ms": round(statistics.median(samples), 3),
        "p95_ms": round(float(np.percentile(samples, 95)), 3),
        "min_ms": round(min(samples), 3),
        "n": repeats,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, choices=("cpu", "mps", "cuda"))
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    import torch
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    device = args.device or (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    train = pd.read_parquet(DATA_DIR / "train_n9000_seed0.parquet")
    one = [test.text.iloc[0]]
    batch = list(test.text.iloc[: args.batch])

    print(f"machine: {platform.processor() or platform.machine()} · torch device {device}")
    print(f"measuring {args.repeats} repeats per configuration, batch={args.batch}")
    print("NOTE: run this on an otherwise idle machine. A concurrent CPU-saturating\n"
          "      job inflates every number here, and a latency figure measured under\n"
          "      contention is exactly the kind of unverifiable claim this repo is\n"
          "      currently walking back.\n")

    load = _load_average()
    if load is not None and load > 2.0:
        print(f"      !! 1-minute load average is {load:.1f} — results will be pessimistic.\n")

    results: dict = {}

    # -- TF-IDF: CPU only, no accelerator anywhere in the path ---------------
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(train.text)
    clf = LogisticRegression(C=10.0, max_iter=2000).fit(Xtr, train.label)
    results["tfidf"] = {
        "single": timed(lambda: clf.predict(vec.transform(one)), args.repeats),
        "batch": timed(lambda: clf.predict(vec.transform(batch)), args.repeats),
        "accelerator": "none (CPU)",
        "params_m": 0,
    }

    # -- frozen transformer encoders ----------------------------------------
    for name, model_id, n_params in [
        ("minilm", "sentence-transformers/all-MiniLM-L6-v2", 22),
        ("distilbert", "distilbert-base-uncased", 66),
    ]:
        tok = AutoTokenizer.from_pretrained(model_id)
        enc = AutoModel.from_pretrained(model_id).eval().to(device)

        def embed(texts, tok=tok, enc=enc):
            with torch.no_grad():
                b = tok(texts, padding=True, truncation=True, max_length=128,
                        return_tensors="pt").to(device)
                h = enc(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).float()
                p = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                return torch.nn.functional.normalize(p, dim=1).cpu().numpy()

        head = LogisticRegression(C=10.0, max_iter=2000).fit(embed(list(train.text[:2000])),
                                                             train.label[:2000])
        results[name] = {
            "single": timed(lambda: head.predict(embed(one)), args.repeats),
            "batch": timed(lambda: head.predict(embed(batch)), args.repeats),
            "accelerator": device,
            "params_m": n_params,
        }

    # -- the fine-tuned classifier ------------------------------------------
    if CKPT.exists():
        tok = AutoTokenizer.from_pretrained(CKPT)
        ft = AutoModelForSequenceClassification.from_pretrained(CKPT).eval().to(device)

        def classify(texts):
            with torch.no_grad():
                b = tok(texts, padding=True, truncation=True, max_length=128,
                        return_tensors="pt").to(device)
                return ft(**b).logits.argmax(-1).cpu().numpy()

        results["finetuned"] = {
            "single": timed(lambda: classify(one), args.repeats),
            "batch": timed(lambda: classify(batch), args.repeats),
            "accelerator": device,
            "params_m": 66,
        }

    # -- derive throughput and cost -----------------------------------------
    print(f"{'arm':<12}{'p50 batch=1':>14}{'p95 batch=1':>14}{'queries/sec':>14}"
          f"{'$/1M queries':>15}{'hardware':>12}")
    for name, r in results.items():
        per_query_ms = r["batch"]["p50_ms"] / args.batch
        qps = 1000.0 / per_query_ms
        hourly = CPU_HOURLY if r["accelerator"] == "none (CPU)" else GPU_HOURLY
        usd_per_query = hourly / 3600 / qps
        r.update({
            "queries_per_second_batched": round(qps, 1),
            "usd_per_query_at_full_utilisation": usd_per_query,
            "usd_per_1m_queries": round(usd_per_query * 1e6, 4),
            "instance_usd_per_hour": hourly,
            "monthly_fixed_usd": round(hourly * 24 * 30, 2),
        })
        print(f"{name:<12}{r['single']['p50_ms']:>13.1f}m{r['single']['p95_ms']:>13.1f}m"
              f"{qps:>14.0f}{r['usd_per_1m_queries']:>15.4f}"
              f"{'CPU' if hourly == CPU_HOURLY else 'GPU':>12}")

    print(f"\nAPI arm for comparison: ${API_USD_PER_QUERY_UNCACHED * 1e6:,.0f} per 1M queries "
          f"(estimated tokens, list price {PRICES_RETRIEVED}).")

    print("\nDeliberately NOT reporting an 'N times cheaper' ratio. At full utilisation the")
    print("fixed cost amortises over so many queries that the ratio runs into the millions,")
    print("which is arithmetically true and practically meaningless -- it is the same trap")
    print("that produced this repo's four contradictory cost figures. What a practitioner")
    print("can act on is the fixed monthly cost and the volume at which it pays for itself:")
    print(f"\n  {'arm':<12}{'fixed $/mo':>12}{'break-even vs API':>22}{'hardware':>11}")
    for name, r in results.items():
        be = r["monthly_fixed_usd"] / API_USD_PER_QUERY_UNCACHED
        r["breakeven_queries_per_month"] = int(be)
        print(f"  {name:<12}{r['monthly_fixed_usd']:>12,.0f}{be:>16,.0f} q/mo"
              f"{'CPU' if r['instance_usd_per_hour'] == CPU_HOURLY else 'GPU':>11}")
    print("\n  The gap between the CPU row and the GPU rows is the finding: a bag of character")
    print("  n-grams needs no accelerator at all, so it pays for itself an order of magnitude")
    print("  sooner than anything that needs a GPU resident.")

    payload = {
        "measured_on": {
            "machine": platform.processor() or platform.machine(),
            "platform": platform.platform(),
            "torch_device": device,
            "batch_size": args.batch,
            "repeats": args.repeats,
        },
        "prices_retrieved": PRICES_RETRIEVED,
        "prices_usd_per_hour": {"gpu": GPU_HOURLY, "cpu": CPU_HOURLY},
        "api_usd_per_query_uncached": API_USD_PER_QUERY_UNCACHED,
        "api_note": "ESTIMATED from ~1500 input + ~50 output tokens. Not measured. "
                    "The rebuilt harness records usage per call and replaces this.",
        "arms": results,
        "load_average_at_start": load,
        "caveats": [
            "Latency is measured on the machine named above, not on the T4 the "
            "published cost model priced. Throughput on other hardware will differ.",
            "Measurements are only valid on an idle machine; load_average_at_start "
            "is recorded so a contended run can be spotted and discarded.",
            "Instance prices are list quotes, not invoices.",
            "Utilisation is not assumed here: $/query is quoted at FULL utilisation, "
            "and the fixed monthly cost is given separately so the reader can apply "
            "their own load. The published model folded a 10% assumption into a "
            "single ratio.",
            "Training cost is excluded from $/query for every arm. It is not "
            "symmetric: the frozen arms fit a logistic regression in seconds on CPU, "
            "the fine-tuned arm needs a GPU sweep.",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
