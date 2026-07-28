"""Call assembly and execution: synchronous (for latency) and Batch API (for quality).

Two execution paths, deliberately:

  run_batch  50% cheaper, sidesteps rate limits, and is how every quality number
             is produced. Gives no usable latency signal -- requests sit in a queue.
  run_sync   sequential, one call at a time, wall-clock timed. The only way to get
             p50/p95 per-query latency, which the repo has never actually measured.

Ordering is a parameter, not a property of how the file happened to be sorted.
`blocked` reproduces the v1 collection error (test set fed in class order); with
queries_per_call > 1 that is the ablation. With queries_per_call == 1 the two
orders should agree to within run-to-run noise -- that agreement is the control
showing per-query evaluation is order-invariant.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from .prompting import PromptConfig, TfidfRetriever, build_call, max_tokens_for

ORDERS = ("blocked", "shuffled")


@dataclass
class CallRequest:
    """One API call: which rows it covers, and the exact body that will be sent."""

    custom_id: str
    row_ids: list[str]
    body: dict


def order_rows(rows: pd.DataFrame, order: str, shuffle_seed: int) -> pd.DataFrame:
    """Return `rows` in evaluation order. Deterministic given (order, shuffle_seed)."""
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
    if order == "blocked":
        return rows.sort_values(["label", "row_id"], kind="stable").reset_index(drop=True)
    rng = np.random.default_rng(shuffle_seed)
    return rows.iloc[rng.permutation(len(rows))].reset_index(drop=True)


def assemble_calls(
    label_names: tuple[str, ...],
    rows: pd.DataFrame,
    cfg: PromptConfig,
    order: str,
    shuffle_seed: int,
    glosses: dict[str, str] | None = None,
    fixed_demos: pd.DataFrame | None = None,
    retriever: TfidfRetriever | None = None,
) -> list[CallRequest]:
    """Split ordered rows into calls of `cfg.queries_per_call` and build each body."""
    if cfg.is_retrieved:
        if retriever is None:
            raise ValueError("retrieved_kshot needs a retriever")
        if cfg.queries_per_call != 1:
            raise ValueError(
                "retrieved_kshot retrieves demonstrations per query, so it requires "
                "queries_per_call=1"
            )
    ordered = order_rows(rows, order, shuffle_seed)
    calls: list[CallRequest] = []
    for start in range(0, len(ordered), cfg.queries_per_call):
        chunk = ordered.iloc[start : start + cfg.queries_per_call]
        demos = retriever(chunk.iloc[0].text) if retriever is not None else None
        body = build_call(cfg, label_names, chunk, glosses, fixed_demos, demos)
        calls.append(
            CallRequest(
                custom_id=f"call-{start // cfg.queries_per_call:05d}",
                row_ids=list(chunk.row_id),
                body=body,
            )
        )
    return calls


def request_params(body: dict, model: str, cfg: PromptConfig) -> dict:
    """Attach model-level parameters to a prompt body.

    No `temperature` / `top_p` / `top_k`: they are rejected outright on Opus 4.7+,
    Opus 5, Sonnet 5 and Fable 5, and on the models that still accept them they
    would not buy determinism anyway. Run-to-run variance is measured instead
    (that is what --runs is for), not assumed away.
    """
    output_config = dict(body["output_config"])
    if cfg.effort:
        output_config["effort"] = cfg.effort
    return {
        "model": model,
        "max_tokens": max_tokens_for(cfg.queries_per_call),
        "system": body["system"],
        "messages": body["messages"],
        "output_config": output_config,
        "thinking": {"type": cfg.thinking},
    }


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------


def _usage_dict(usage) -> dict:
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
    }


def parse_message(message, call: CallRequest, queries_per_call: int) -> list[dict]:
    """Turn one API response into one prediction row per row_id in the call.

    A row whose prediction is missing gets `pred_intent = None` and a stated
    reason. Nothing is silently dropped and nothing is guessed -- a null here is
    a measurement, and the scorer must decide how to treat it rather than never
    learning it happened.
    """
    base = {
        "stop_reason": getattr(message, "stop_reason", None),
        "resolved_model": getattr(message, "model", None),
        **_usage_dict(getattr(message, "usage", None)),
    }

    def blank(row_id: str, position: int, error: str) -> dict:
        return {"row_id": row_id, "position_in_call": position, "pred_intent": None,
                "error": error, **base}

    if getattr(message, "stop_reason", None) == "refusal":
        return [blank(rid, i, "refusal") for i, rid in enumerate(call.row_ids)]

    text = next((b.text for b in message.content if b.type == "text"), None)
    if text is None:
        return [blank(rid, i, "no text block") for i, rid in enumerate(call.row_ids)]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        # Should be unreachable under output_config.format; recorded, not repaired.
        return [blank(rid, i, f"invalid JSON: {exc}") for i, rid in enumerate(call.row_ids)]

    if queries_per_call == 1:
        return [{"row_id": call.row_ids[0], "position_in_call": 0,
                 "pred_intent": payload.get("intent"), "error": None, **base}]

    by_id = {}
    for item in payload.get("predictions", []):
        idx = item.get("id")
        if isinstance(idx, int) and 1 <= idx <= len(call.row_ids) and idx not in by_id:
            by_id[idx] = item.get("intent")
    return [
        {"row_id": rid, "position_in_call": i, "pred_intent": by_id.get(i + 1),
         "error": None if (i + 1) in by_id else "missing from batched reply", **base}
        for i, rid in enumerate(call.row_ids)
    ]


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


def run_sync(
    client,
    calls: list[CallRequest],
    model: str,
    cfg: PromptConfig,
    warmup: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Sequential execution with per-call wall-clock timing.

    One warmup call is issued and discarded first: a new response schema is
    compiled on first use and the cache prefix is cold, so including that call
    would inflate p50 with a one-off cost the deployment does not pay per query.
    """
    if warmup and calls:
        client.messages.create(**request_params(calls[0].body, model, cfg))

    rows: list[dict] = []
    for i, call in enumerate(calls):
        started = time.perf_counter()
        message = client.messages.create(**request_params(call.body, model, cfg))
        elapsed_ms = (time.perf_counter() - started) * 1000
        for row in parse_message(message, call, cfg.queries_per_call):
            rows.append({**row, "call_id": call.custom_id, "latency_ms": elapsed_ms})
        if progress:
            progress(i + 1, len(calls))
    return rows


def run_batch(
    client,
    calls: list[CallRequest],
    model: str,
    cfg: PromptConfig,
    poll_seconds: int = 60,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict], str]:
    """Submit via the Batch API and block until results land. Returns (rows, batch_id).

    Results come back in arbitrary order, so they are keyed by `custom_id` --
    never by position.
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    batch = client.messages.batches.create(
        requests=[
            Request(
                custom_id=call.custom_id,
                params=MessageCreateParamsNonStreaming(
                    **request_params(call.body, model, cfg)
                ),
            )
            for call in calls
        ]
    )
    if progress:
        progress(f"submitted batch {batch.id} ({len(calls)} calls)")

    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        if progress:
            counts = batch.request_counts
            progress(
                f"{batch.processing_status}: {counts.succeeded} succeeded, "
                f"{counts.processing} processing, {counts.errored} errored"
            )
        time.sleep(poll_seconds)

    by_id = {call.custom_id: call for call in calls}
    rows: list[dict] = []
    seen: set[str] = set()
    for result in client.messages.batches.results(batch.id):
        call = by_id[result.custom_id]
        seen.add(result.custom_id)
        if result.result.type == "succeeded":
            parsed = parse_message(result.result.message, call, cfg.queries_per_call)
        else:
            reason = f"batch result {result.result.type}"
            parsed = [
                {"row_id": rid, "position_in_call": i, "pred_intent": None, "error": reason,
                 "stop_reason": None, "resolved_model": None, "input_tokens": None,
                 "output_tokens": None, "cache_creation_input_tokens": None,
                 "cache_read_input_tokens": None}
                for i, rid in enumerate(call.row_ids)
            ]
        rows += [{**row, "call_id": call.custom_id, "latency_ms": None} for row in parsed]

    for custom_id in set(by_id) - seen:
        call = by_id[custom_id]
        rows += [
            {"row_id": rid, "position_in_call": i, "pred_intent": None,
             "error": "no batch result returned", "stop_reason": None,
             "resolved_model": None, "input_tokens": None, "output_tokens": None,
             "cache_creation_input_tokens": None, "cache_read_input_tokens": None,
             "call_id": custom_id, "latency_ms": None}
            for i, rid in enumerate(call.row_ids)
        ]
    return rows, batch.id
