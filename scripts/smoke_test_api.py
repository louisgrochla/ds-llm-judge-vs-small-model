"""Confirm the pinned model snapshot resolves and the harness's API assumptions hold.

HANDOVER §6.1: "Confirm the snapshot resolves before planning around it -- if the
model is not servable, this is a new study, not a repair." §7 puts this on 28 July.

Costs a few cents: four short calls plus two metadata reads.

    export ANTHROPIC_API_KEY=...        # or: ant auth login
    python scripts/smoke_test_api.py --model claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.harness import datasets, prompting, runner  # noqa: E402

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--cheap-model", default="claude-haiku-4-5")
    ap.add_argument("--dataset", default="banking77")
    args = ap.parse_args()

    import anthropic

    client = anthropic.Anthropic()
    spec = datasets.get_dataset(args.dataset)
    rows = spec.load_split("dev").head(1)
    results: list[tuple[str, str, str]] = []

    def check(name: str, fn) -> None:
        try:
            status, detail = fn()
        except Exception as exc:  # noqa: BLE001 -- a failed check is the finding
            status, detail = FAIL, f"{type(exc).__name__}: {exc}"
        results.append((status, name, detail))
        print(f"[{status}] {name}\n       {detail}")

    def snapshot_resolves(model: str):
        m = client.models.retrieve(model)
        return PASS, f"{m.id} ({m.display_name}), context {m.max_input_tokens:,}, max out {m.max_tokens:,}"

    check(f"model snapshot resolves: {args.model}", lambda: snapshot_resolves(args.model))
    check(f"cheap tier resolves: {args.cheap_model}", lambda: snapshot_resolves(args.cheap_model))

    cfg = prompting.PromptConfig(rung="zero_shot", queries_per_call=1)
    calls = runner.assemble_calls(spec.label_names, rows, cfg, "shuffled", 0)
    params = runner.request_params(calls[0].body, args.model, cfg)

    state: dict = {}

    def structured_output():
        started = time.perf_counter()
        msg = client.messages.create(**params)
        state["latency_ms"] = (time.perf_counter() - started) * 1000
        state["message"] = msg
        parsed = runner.parse_message(msg, calls[0], 1)[0]
        pred = parsed["pred_intent"]
        if pred is None:
            return FAIL, f"no prediction: {parsed['error']}"
        if pred not in spec.label_names:
            return FAIL, f"returned an out-of-vocabulary label: {pred!r}"
        return PASS, (
            f'query "{rows.iloc[0].text[:48]}..." -> {pred} '
            f"(gold {rows.iloc[0].intent}), served by {msg.model}"
        )

    check(f"structured output, closed {spec.n_classes}-value enum", structured_output)

    def usage_captured():
        u = state["message"].usage
        got = {k: getattr(u, k, None) for k in
               ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
        if got["input_tokens"] is None or got["output_tokens"] is None:
            return FAIL, f"usage incomplete: {got}"
        return PASS, json.dumps(got)

    check("per-call usage recorded", usage_captured)
    check("synchronous latency measurable",
          lambda: (PASS, f"{state['latency_ms']:.0f} ms cold (schema compile + cold cache)"))

    def cache_prefix():
        msg = client.messages.create(**params)
        read = getattr(msg.usage, "cache_read_input_tokens", 0) or 0
        if read > 0:
            return PASS, f"second identical call read {read} cached prefix tokens"
        return WARN, (
            "no cache read on the second call -- the zero-shot prefix is likely below "
            "this model's minimum cacheable length. Expected to cache once glosses or "
            "k-shot demos are added; harmless either way, but it means the cost model "
            "must not assume cache hits for the zero-shot rung."
        )

    check("prompt prefix caches", cache_prefix)

    def batch_reachable():
        page = client.messages.batches.list(limit=1)
        return PASS, f"Batch API reachable ({len(page.data)} recent batch(es) visible)"

    check("Batch API reachable", batch_reachable)

    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    n_warn = sum(1 for s, _, _ in results if s == WARN)
    print(f"\n{len(results) - n_fail - n_warn} passed, {n_warn} warned, {n_fail} failed")
    if n_fail:
        print("\nA failed snapshot check means the study design in HANDOVER §6.1 needs "
              "revisiting before any run is launched -- do not work around it.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
