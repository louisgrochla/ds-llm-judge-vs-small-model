"""Generate the n=9000 training subsets that notebook 01 hadn't produced.

Reproduces the same stratified_shuffle logic from notebook 01 section 7 against
the same train_pool (after val + dev carve-out). Deterministic per seed — the
n=50/100/.../5000 subsets that already exist on disk are bit-for-bit identical
to what this script would produce, so we only write the n=9000 ones.

Run: .venv/bin/python scripts/generate_n9000_subsets.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROCESSED_DIR = Path(__file__).resolve().parent.parent / 'data' / 'processed'

# Reconstruct the train_pool the same way notebook 01 did
VAL_SIZE = 500
DEV_SIZE = 300
RANDOM_SEED = 42
N_SEEDS = 5
N = 9000

# Load the full HF Banking77 train split — same source as notebook 01
import json
from huggingface_hub import hf_hub_download

REPO = 'PolyAI/banking77'
train_path = hf_hub_download(REPO, 'default/train/0000.parquet', revision='refs/convert/parquet', repo_type='dataset')
train_df = pd.read_parquet(train_path)

# Carve val (matches notebook 01 section 5)
train_pool_after_val, _ = train_test_split(
    train_df, test_size=VAL_SIZE, stratify=train_df['label'], random_state=RANDOM_SEED,
)
train_pool_after_val = train_pool_after_val.reset_index(drop=True)

# Carve dev (matches notebook 01 section 6)
train_pool, _ = train_test_split(
    train_pool_after_val, test_size=DEV_SIZE,
    stratify=train_pool_after_val['label'], random_state=RANDOM_SEED,
)
train_pool = train_pool.reset_index(drop=True)

print(f'train_pool size: {len(train_pool)} (need at least {N})')


def stratified_shuffle(df, label_col, seed):
    """Same function as notebook 01 section 7."""
    rng = np.random.default_rng(seed)
    by_label = {}
    for lbl in df[label_col].unique():
        idx = np.array(df.index[df[label_col] == lbl])
        rng.shuffle(idx)
        by_label[int(lbl)] = idx

    label_order = sorted(by_label.keys())
    rng.shuffle(label_order)

    result = []
    pointers = {lbl: 0 for lbl in by_label}
    while any(pointers[lbl] < len(by_label[lbl]) for lbl in label_order):
        for lbl in label_order:
            if pointers[lbl] < len(by_label[lbl]):
                result.append(int(by_label[lbl][pointers[lbl]]))
                pointers[lbl] += 1
    return np.array(result)


for seed in range(N_SEEDS):
    shuffled_idx = stratified_shuffle(train_pool, 'label', seed)
    shuffled_pool = train_pool.loc[shuffled_idx].reset_index(drop=True)
    subset = shuffled_pool.iloc[:N].copy()
    classes_present = subset['label'].nunique()
    path = PROCESSED_DIR / f'train_n{N}_seed{seed}.parquet'
    subset.reset_index(drop=True).to_parquet(path)
    print(f'  n={N} seed={seed}: {len(subset)} rows, {classes_present}/77 classes — saved {path.name}')
