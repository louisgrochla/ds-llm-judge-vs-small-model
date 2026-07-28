"""Dataset-agnostic LLM evaluation harness (HANDOVER §6.1).

Rebuild of the broken v1 baseline. Design constraints, all load-bearing:

  * one query per API call by default -- matches the deployment the cost model prices
  * structured outputs with a closed label enum -- an invalid label is impossible,
    so the tolerant regex parser in src/llm_eval.py is retired
  * blocked-vs-shuffled ordering is a first-class flag, not an afterthought
  * per-call `usage` and wall-clock latency are recorded, so cost is measured
  * every run writes a manifest pinning model, prompt, schema, seeds and git sha

Nothing here knows about Banking77 specifically; adding CLINC150 (§6.3) is one
entry in `datasets.REGISTRY`.
"""

from .datasets import REGISTRY, DatasetSpec, get_dataset
from .prompting import PromptConfig, RUNGS, build_call, response_schema
from .runner import CallRequest, run_batch, run_sync

__all__ = [
    "REGISTRY",
    "DatasetSpec",
    "get_dataset",
    "PromptConfig",
    "RUNGS",
    "build_call",
    "response_schema",
    "CallRequest",
    "run_batch",
    "run_sync",
]
