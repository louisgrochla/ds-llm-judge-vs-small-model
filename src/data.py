"""Processed data loaders. All splits live in data/processed/ as parquet files written by notebook 01."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

TRAINING_SIZES = [50, 100, 250, 500, 1000, 2500, 5000, 9000]
N_SEEDS = 5
DEV_SLICE_SIZE = 300
VAL_SET_SIZE = 500


def load_test_set() -> pd.DataFrame:
    """Banking77 official 3,080-row test split. Used for the final LLM eval and every DistilBERT eval."""
    return pd.read_parquet(DATA_DIR / "test.parquet")


def load_dev_slice() -> pd.DataFrame:
    """300-row stratified slice carved from the train pool. Used for LLM prompt iteration in notebook 02."""
    return pd.read_parquet(DATA_DIR / "dev_slice.parquet")


def load_val_set() -> pd.DataFrame:
    """500-row fixed val set carved from the train pool. Same val for every n / every seed in notebook 03."""
    return pd.read_parquet(DATA_DIR / "val.parquet")


def load_train_subset(n: int, seed: int = 0) -> pd.DataFrame:
    """Stratified training subset of size n for one of the 5 random seeds."""
    if n not in TRAINING_SIZES:
        raise ValueError(f"n must be in {TRAINING_SIZES}, got {n}")
    if not 0 <= seed < N_SEEDS:
        raise ValueError(f"seed must be in [0, {N_SEEDS}), got {seed}")
    return pd.read_parquet(DATA_DIR / f"train_n{n}_seed{seed}.parquet")


def load_label_names() -> list[str]:
    """77 Banking77 intent names, in order matching the integer label IDs."""
    return (DATA_DIR / "label_names.txt").read_text().splitlines()
