"""Offline checks on the eval harness. No API key, no network, no spend.

These assert the properties that the v1 baseline turned out not to have --
above all that `shuffled` really is shuffled and that every test row is covered
exactly once. `scripts/verify_handover_claims.py` proves v1 was broken; this
proves the replacement is not, before a single token is bought.

    python scripts/selftest_harness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.harness import artifacts, datasets, prompting, runner  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'ok  ' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def main() -> int:
    spec = datasets.get_dataset("banking77")
    test = spec.load_split("test")

    check("test split loads", len(test) == 3080, f"{len(test)} rows")
    check("label vocabulary is 77", spec.n_classes == 77)
    check("row_ids unique", test.row_id.is_unique)
    check("intent column matches label ids",
          all(spec.label_names[r.label] == r.intent for r in test.itertuples()))

    # -- ordering: the v1 defect, made visible ------------------------------
    def transitions(df: pd.DataFrame) -> int:
        v = df.intent.to_numpy()
        return int((v[1:] != v[:-1]).sum())

    blocked = runner.order_rows(test, "blocked", 0)
    shuffled = runner.order_rows(test, "shuffled", 20260729)
    expected_random = len(test) * (1 - 1 / spec.n_classes)
    check("blocked order reproduces the v1 signature",
          transitions(blocked) == spec.n_classes - 1,
          f"{transitions(blocked)} transitions (v1 artifact: 76)")
    check("shuffled order is genuinely shuffled",
          transitions(shuffled) > 0.95 * expected_random,
          f"{transitions(shuffled)} transitions (random ~{expected_random:.0f})")
    check("shuffle is reproducible from the seed",
          runner.order_rows(test, "shuffled", 20260729).row_id.equals(shuffled.row_id))
    check("ordering is a permutation, not a filter",
          set(blocked.row_id) == set(shuffled.row_id) == set(test.row_id))

    # -- call assembly ------------------------------------------------------
    for qpc in (1, 150):
        cfg = prompting.PromptConfig(rung="zero_shot", queries_per_call=qpc)
        calls = runner.assemble_calls(spec.label_names, test, cfg, "shuffled", 20260729)
        covered = [rid for c in calls for rid in c.row_ids]
        check(f"queries_per_call={qpc}: every row covered exactly once",
              len(covered) == len(test) and len(set(covered)) == len(test),
              f"{len(calls)} calls")
        check(f"queries_per_call={qpc}: system prompt is byte-identical across calls",
              len({c.body["system"][0]["text"] for c in calls}) == 1)

    # -- schema -------------------------------------------------------------
    single = prompting.response_schema(spec.label_names, 1)
    batched = prompting.response_schema(spec.label_names, 150)
    check("single-query schema closes the label set",
          single["properties"]["intent"]["enum"] == list(spec.label_names))
    check("batched schema closes the label set",
          batched["properties"]["predictions"]["items"]["properties"]["intent"]["enum"]
          == list(spec.label_names))
    check("schema forbids extra keys", single["additionalProperties"] is False)
    check("max_tokens scales with batch size",
          prompting.max_tokens_for(1) < prompting.max_tokens_for(150) <= 16000)

    # -- config guards ------------------------------------------------------
    def raises(fn) -> bool:
        try:
            fn()
        except (ValueError, KeyError):
            return True
        return False

    check("k-shot rung rejects k=0", raises(lambda: prompting.PromptConfig(rung="fixed_kshot", k=0)))
    check("zero-shot rung rejects demonstrations",
          raises(lambda: prompting.PromptConfig(rung="zero_shot", k=2)))
    check("unknown rung rejected", raises(lambda: prompting.PromptConfig(rung="chain_of_thought")))
    check("unknown order rejected", raises(lambda: runner.order_rows(test, "sorted", 0)))
    check("retrieved rung refuses batched calls",
          raises(lambda: runner.assemble_calls(
              spec.label_names, test.head(4),
              prompting.PromptConfig(rung="retrieved_kshot", k=2, queries_per_call=2),
              "shuffled", 0, retriever=object())))

    # -- parsing ------------------------------------------------------------
    class Block:
        type = "text"

        def __init__(self, text):
            self.text = text

    class Msg:
        def __init__(self, text, stop_reason="end_turn"):
            self.content = [Block(text)]
            self.stop_reason = stop_reason
            self.model = "test-model"
            self.usage = type("U", (), {"input_tokens": 10, "output_tokens": 5,
                                        "cache_creation_input_tokens": 0,
                                        "cache_read_input_tokens": 0})()

    call1 = runner.CallRequest("call-0", ["test:0"], {})
    ok = runner.parse_message(Msg('{"intent": "card_arrival"}'), call1, 1)
    check("single-query reply parses", ok[0]["pred_intent"] == "card_arrival")

    refused = runner.parse_message(Msg("", "refusal"), call1, 1)
    check("refusal yields a null prediction, not a guess",
          refused[0]["pred_intent"] is None and refused[0]["error"] == "refusal")

    call3 = runner.CallRequest("call-0", ["test:0", "test:1", "test:2"], {})
    partial = runner.parse_message(
        Msg('{"predictions": [{"id": 1, "intent": "age_limit"}, {"id": 3, "intent": "atm_support"}]}'),
        call3, 3)
    check("short batched reply is recorded, not silently padded",
          [r["pred_intent"] for r in partial] == ["age_limit", None, "atm_support"]
          and partial[1]["error"] == "missing from batched reply")

    # -- cost accounting ----------------------------------------------------
    preds = pd.DataFrame({
        "run_index": [0, 0, 0], "call_id": ["call-0"] * 3,
        "input_tokens": [1_000_000] * 3, "output_tokens": [1_000_000] * 3,
        "cache_read_input_tokens": [0] * 3, "cache_creation_input_tokens": [0] * 3,
    })
    sync_cost = artifacts.cost_usd(preds, "claude-sonnet-4-6", "sync")
    batch_cost = artifacts.cost_usd(preds, "claude-sonnet-4-6", "batch")
    check("one batched call is billed once, not once per row",
          sync_cost["input_tokens"] == 1_000_000, f"${sync_cost['total_usd']}")
    check("batch discount is applied",
          abs(batch_cost["total_usd"] - sync_cost["total_usd"] / 2) < 1e-6,
          f"${batch_cost['total_usd']} vs ${sync_cost['total_usd']}")
    check("unpriced model refuses to invent a rate",
          "error" in artifacts.cost_usd(preds, "some-future-model", "sync"))

    # -- latency and variance ----------------------------------------------
    timed = pd.DataFrame({
        "run_index": [0] * 5, "call_id": [f"call-{i}" for i in range(5)],
        "latency_ms": [100.0, 200.0, 300.0, 400.0, 5000.0],
    })
    lat = artifacts.latency_summary(timed)
    check("latency reports percentiles, not a mean",
          lat["p50_ms"] == 300.0 and "mean_ms" not in lat, f"p50 {lat['p50_ms']}, p95 {lat['p95_ms']}")

    varied = pd.DataFrame({
        "row_id": ["a", "b", "a", "b"], "run_index": [0, 0, 1, 1],
        "pred_intent": ["x", "y", "x", "z"],
    })
    var = artifacts.run_variance(varied)
    check("run-to-run disagreement is measured",
          var["disagreement_fraction"] == 0.5, f"{var['disagreement_fraction']}")

    # -- optional: gloss file, and fingerprint distinctness across rungs -----
    try:
        glosses = spec.load_glosses()
        check("gloss file covers every label", len(glosses) == spec.n_classes)

        pool = spec.load_split(spec.demo_pool_split)
        sample = test.head(2)
        prints = {}
        for cfg, kwargs in [
            (prompting.PromptConfig("zero_shot"), {}),
            (prompting.PromptConfig("glosses"), {"glosses": glosses}),
            (prompting.PromptConfig("fixed_kshot", k=1),
             {"glosses": glosses,
              "fixed_demos": prompting.fixed_demos(pool, spec.label_names, 1, 0)}),
            (prompting.PromptConfig("retrieved_kshot", k=2),
             {"glosses": glosses, "retriever": prompting.TfidfRetriever(pool, 2)}),
        ]:
            calls = runner.assemble_calls(
                spec.label_names, sample, cfg, "shuffled", 0, **kwargs)
            prints[cfg.rung] = prompting.prompt_fingerprint(calls[0].body, cfg)
        check("every rung has a distinct prompt fingerprint",
              len(set(prints.values())) == len(prints), str(prints))

        retrieved = runner.assemble_calls(
            spec.label_names, sample, prompting.PromptConfig("retrieved_kshot", k=2),
            "shuffled", 0, glosses=glosses, retriever=prompting.TfidfRetriever(pool, 2))
        users = [c.body["messages"][0]["content"] for c in retrieved]
        check("retrieved demos land in the user turn, not the cached prefix",
              all("Similar labelled examples:" in u for u in users)
              and users[0] != users[1])
        check("retrieved rung keeps the cached prefix byte-identical",
              len({c.body["system"][0]["text"] for c in retrieved}) == 1)
    except FileNotFoundError:
        print(f"[skip] gloss file absent ({spec.glosses_path.name}) -- "
              "zero_shot runs; glosses/k-shot rungs are blocked until it exists")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all harness self-checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
