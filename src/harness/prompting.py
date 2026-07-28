"""The prompt ladder (HANDOVER §6.1) and the closed-enum response schema.

Four rungs, each strictly additive over the last:

    zero_shot        instruction + the label vocabulary
    glosses          + a one-line description per label
    fixed_kshot      + k demonstrations per class, identical for every query
    retrieved_kshot  + k demonstrations retrieved per query from the demo pool

The ladder is run on the dev slice and the winner is frozen before test is
touched. Everything stable across queries goes in `system` and is cached; only
the query (and, for the retrieved rung, its own demonstrations) goes in the user
turn, after the cache breakpoint.

Demonstrations are labelled supervision. `n_demonstrations` is recorded in the
run manifest so the re-analysis can charge them to the label budget.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

RUNGS = ("zero_shot", "glosses", "fixed_kshot", "retrieved_kshot")

_TASK = (
    "You are an intent classifier for a {domain} assistant.\n"
    "Assign each user query to exactly one intent from the closed list below.\n"
    "Choose the single best match even when several intents look plausible."
)


@dataclass(frozen=True)
class PromptConfig:
    """One rung of the ladder, plus the knobs that change what a call looks like."""

    rung: str = "zero_shot"
    k: int = 0
    queries_per_call: int = 1
    domain: str = "banking customer-service"
    thinking: str = "disabled"
    effort: str | None = None
    extra_instruction: str = ""

    def __post_init__(self) -> None:
        if self.rung not in RUNGS:
            raise ValueError(f"rung must be one of {RUNGS}, got {self.rung!r}")
        if self.rung in ("fixed_kshot", "retrieved_kshot") and self.k < 1:
            raise ValueError(f"rung {self.rung!r} needs k >= 1, got {self.k}")
        if self.rung in ("zero_shot", "glosses") and self.k:
            raise ValueError(f"rung {self.rung!r} takes no demonstrations, got k={self.k}")
        if self.queries_per_call < 1:
            raise ValueError("queries_per_call must be >= 1")
        if self.thinking not in ("disabled", "adaptive"):
            raise ValueError("thinking must be 'disabled' or 'adaptive'")

    @property
    def uses_glosses(self) -> bool:
        return self.rung != "zero_shot"

    @property
    def is_retrieved(self) -> bool:
        return self.rung == "retrieved_kshot"


def response_schema(label_names: tuple[str, ...], queries_per_call: int) -> dict:
    """JSON schema constraining the reply to labels that exist.

    The 77-value `enum` is what makes an out-of-vocabulary prediction impossible
    by construction. Note the API does not enforce array length, so the runner
    still checks the batched form returned one object per query.
    """
    one = {
        "type": "object",
        "properties": {"intent": {"type": "string", "enum": list(label_names)}},
        "required": ["intent"],
        "additionalProperties": False,
    }
    if queries_per_call == 1:
        return one
    item = {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "intent": {"type": "string", "enum": list(label_names)},
        },
        "required": ["id", "intent"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"predictions": {"type": "array", "items": item}},
        "required": ["predictions"],
        "additionalProperties": False,
    }


def _label_block(label_names: tuple[str, ...], glosses: dict[str, str] | None) -> str:
    if glosses is None:
        return "\n".join(f"- {name}" for name in label_names)
    return "\n".join(f"- {name}: {glosses[name]}" for name in label_names)


def _demo_block(demos: pd.DataFrame) -> str:
    return "\n".join(f'query: "{r.text}"\nintent: {r.intent}' for r in demos.itertuples())


def build_system(
    cfg: PromptConfig,
    label_names: tuple[str, ...],
    glosses: dict[str, str] | None = None,
    fixed_demos: pd.DataFrame | None = None,
) -> list[dict]:
    """The stable, cacheable half of the prompt. Byte-identical across every call."""
    parts = [_TASK.format(domain=cfg.domain), "", "Intents:", _label_block(label_names, glosses)]
    if fixed_demos is not None and len(fixed_demos):
        parts += ["", "Labelled examples:", _demo_block(fixed_demos)]
    if cfg.extra_instruction:
        parts += ["", cfg.extra_instruction]
    text = "\n".join(parts)
    # One breakpoint on the only system block: caches the whole prefix. Prefixes
    # below the model's minimum simply will not cache -- usage.cache_read_input_tokens
    # in the artifact says whether it did, so this is measured rather than assumed.
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def build_user(rows: pd.DataFrame, per_query_demos: pd.DataFrame | None = None) -> str:
    """The volatile half: the query (or queries), placed after the cache breakpoint."""
    parts: list[str] = []
    if per_query_demos is not None and len(per_query_demos):
        parts += ["Similar labelled examples:", _demo_block(per_query_demos), ""]
    if len(rows) == 1:
        parts += [f'Classify this query:\n"{rows.iloc[0].text}"']
    else:
        numbered = "\n".join(f'{i + 1}. "{r.text}"' for i, r in enumerate(rows.itertuples()))
        parts += [
            f"Classify each of these {len(rows)} queries. "
            f"Return one object per query, with `id` from 1 to {len(rows)}:",
            numbered,
        ]
    return "\n".join(parts)


def build_call(
    cfg: PromptConfig,
    label_names: tuple[str, ...],
    rows: pd.DataFrame,
    glosses: dict[str, str] | None = None,
    fixed_demos: pd.DataFrame | None = None,
    per_query_demos: pd.DataFrame | None = None,
) -> dict:
    """Assemble the request body for one API call (model/max_tokens added by the runner)."""
    return {
        "system": build_system(cfg, label_names, glosses, fixed_demos),
        "messages": [{"role": "user", "content": build_user(rows, per_query_demos)}],
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": response_schema(label_names, cfg.queries_per_call),
            }
        },
    }


def max_tokens_for(queries_per_call: int) -> int:
    """Enough room for the structured reply and nothing else. Never the default."""
    return min(64 + 48 * queries_per_call, 16000)


# --------------------------------------------------------------------------
# Demonstration selection
# --------------------------------------------------------------------------


def fixed_demos(
    pool: pd.DataFrame, label_names: tuple[str, ...], k: int, seed: int
) -> pd.DataFrame:
    """k demonstrations per class, sampled once and reused for every query."""
    rng = np.random.default_rng(seed)
    picks = []
    for name in label_names:
        cand = pool[pool.intent == name]
        if cand.empty:
            raise ValueError(f"demo pool has no examples of {name!r}")
        take = min(k, len(cand))
        picks.append(cand.iloc[rng.choice(len(cand), size=take, replace=False)])
    return pd.concat(picks).sort_values("intent").reset_index(drop=True)


class TfidfRetriever:
    """Per-query demonstration retrieval over the demo pool.

    Character n-gram TF-IDF, so there is no extra model dependency and no
    embedding-model choice to defend in Setup. The pool is training data only --
    retrieving from test would leak the answer.
    """

    def __init__(self, pool: pd.DataFrame, k: int):
        from sklearn.feature_extraction.text import TfidfVectorizer

        if k < 1:
            raise ValueError("k must be >= 1")
        self.pool = pool.reset_index(drop=True)
        self.k = k
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
        self.matrix = self.vectorizer.fit_transform(self.pool.text)

    def __call__(self, query: str) -> pd.DataFrame:
        sims = (self.vectorizer.transform([query]) @ self.matrix.T).toarray()[0]
        # Ascending order so the closest example sits nearest the query.
        top = np.argsort(-sims)[: self.k][::-1]
        return self.pool.iloc[top].reset_index(drop=True)


def prompt_fingerprint(call_body: dict, cfg: PromptConfig) -> str:
    """SHA-256 pinning 'which prompt produced this number'.

    Covers the config as well as the rendered system block and schema: the
    retrieved rung puts its demonstrations in the user turn to keep the cached
    prefix stable, so hashing `system` alone would collide with the `glosses`
    rung and silently label two different experiments identically.
    """
    payload = json.dumps(
        {
            "system": call_body["system"],
            "output_config": call_body["output_config"],
            "config": asdict(cfg),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
