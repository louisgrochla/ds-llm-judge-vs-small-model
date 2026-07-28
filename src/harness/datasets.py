"""Dataset specs. Everything dataset-specific in the harness lives in this file.

A `DatasetSpec` is the complete contract the rest of the harness needs: a label
vocabulary, named splits, a pool to draw few-shot demonstrations from, and an
optional gloss file. Adding CLINC150 (HANDOVER §6.3) means adding one spec and
the parquet files it points at -- prompting.py and runner.py do not change.

Every row carries a `row_id` of the form "<split>:<position>". Positions are the
parquet's on-disk order, which is fixed, so predictions join back to gold labels
regardless of what order the harness sent them in. This is the join key that the
v1 baseline lacked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
DATA_DIR = DATA_ROOT / "processed"


@dataclass(frozen=True)
class DatasetSpec:
    """One intent-classification dataset, described in harness terms."""

    name: str
    label_names: tuple[str, ...]
    splits: dict[str, str]
    demo_pool_split: str
    data_dir: Path = DATA_DIR
    glosses_path: Path | None = None

    @property
    def n_classes(self) -> int:
        return len(self.label_names)

    def load_split(self, split: str) -> pd.DataFrame:
        """Return columns [row_id, text, label, intent] in on-disk order."""
        if split not in self.splits:
            raise KeyError(f"{self.name} has no split {split!r}; have {sorted(self.splits)}")
        path = self.data_dir / self.splits[split]
        if not path.exists():
            raise FileNotFoundError(f"{self.name}/{split} expected at {path}")
        df = pd.read_parquet(path).reset_index(drop=True)
        missing = {"text", "label"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required column(s) {sorted(missing)}")
        out_of_range = df.label[(df.label < 0) | (df.label >= self.n_classes)]
        if len(out_of_range):
            raise ValueError(
                f"{path} has {len(out_of_range)} label(s) outside [0, {self.n_classes})"
            )
        return pd.DataFrame(
            {
                "row_id": [f"{split}:{i}" for i in range(len(df))],
                "text": df.text.astype(str),
                "label": df.label.astype(int),
                "intent": [self.label_names[i] for i in df.label],
            }
        )

    def load_glosses(self) -> dict[str, str]:
        """One-line description per label, for the `glosses` rung of the prompt ladder.

        Glosses are author-written from the label names alone. They are a prompt
        input, never derived from test rows -- the file records its own provenance.
        """
        if self.glosses_path is None or not self.glosses_path.exists():
            raise FileNotFoundError(
                f"{self.name} has no gloss file at {self.glosses_path}. "
                "The `glosses` and `*_kshot` rungs need one; `zero_shot` does not."
            )
        raw = json.loads(self.glosses_path.read_text())
        glosses = raw["glosses"] if "glosses" in raw else raw
        missing = [c for c in self.label_names if c not in glosses]
        if missing:
            raise ValueError(
                f"{self.glosses_path} is missing {len(missing)} label(s), "
                f"e.g. {missing[:3]}"
            )
        return {c: glosses[c] for c in self.label_names}


CLINC_DIR = DATA_ROOT / "processed_clinc150"


def _labels(directory: Path) -> tuple[str, ...]:
    path = directory / "label_names.txt"
    if not path.exists():
        return ()
    return tuple(line for line in path.read_text().splitlines() if line)


BANKING77 = DatasetSpec(
    name="banking77",
    label_names=_labels(DATA_DIR),
    splits={
        "test": "test.parquet",
        "dev": "dev_slice.parquet",
        "val": "val.parquet",
        # The k-shot demonstration pool. Not the full 10,003-row train set: the
        # largest committed subset. Demonstrations are labelled supervision and
        # are counted in the manifest's `n_demonstrations` (HANDOVER §6.4).
        "demo_pool": "train_n9000_seed0.parquet",
    },
    demo_pool_split="demo_pool",
    data_dir=DATA_DIR,
    glosses_path=Path(__file__).resolve().parent / "glosses" / "banking77.json",
)

# The second taxonomy (HANDOVER §6.3), scoped as mechanism replication. Written by
# `scripts/prepare_clinc150.py`. The out-of-scope class is deliberately absent from
# `label_names`: it has a different sampling density from the 150 in-scope intents,
# so folding it into a macro-F1 would let one anomalous class move the headline.
# Its rows live in oos_test.parquet for the separate out-of-scope analysis.
CLINC150 = DatasetSpec(
    name="clinc150",
    label_names=_labels(CLINC_DIR),
    splits={
        "test": "test.parquet",
        "dev": "dev_slice.parquet",
        "val": "val.parquet",
        "official_val": "official_val.parquet",
        "oos_test": "oos_test.parquet",
        "demo_pool": "demo_pool.parquet",
    },
    demo_pool_split="demo_pool",
    data_dir=CLINC_DIR,
    glosses_path=Path(__file__).resolve().parent / "glosses" / "clinc150.json",
)

REGISTRY: dict[str, DatasetSpec] = {
    spec.name: spec for spec in (BANKING77, CLINC150) if spec.label_names
}


def get_dataset(name: str) -> DatasetSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name]
