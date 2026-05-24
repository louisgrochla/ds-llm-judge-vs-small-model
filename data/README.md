# Data

## Dataset — Banking77 (PolyAI)

**Source:** https://huggingface.co/datasets/PolyAI/banking77
**Paper:** Casanueva et al. (2020), *"Efficient Intent Detection with Dual Sentence Encoders"*, EACL Workshop on NLP for Conversational AI

**Size:** 13,083 customer queries split into 10,003 train / 3,080 test
**Classes:** 77 fine-grained banking intents (e.g. `card_arrival`, `transfer_failed`, `lost_or_stolen_card`, `top_up_failed`, etc.)
**Format:** Tab-separated; one row per query with `text` and `label`

## No download needed — pulled from HuggingFace

The dataset is loaded directly via the `datasets` library, so there's nothing to manually download:

```python
from datasets import load_dataset
ds = load_dataset("PolyAI/banking77")
# ds["train"] — 10,003 rows
# ds["test"]  — 3,080 rows
```

The first call caches data to `~/.cache/huggingface/datasets/`. Subsequent calls are free.

## Data schema

| field | type | example |
|---|---|---|
| `text` | str | "I noticed an extra fee when I withdrew money."  |
| `label` | int (0–76) | 24 |
| `label_text` *(computed)* | str | `transfer_fee_charged` |

The integer-to-name mapping is available via `ds["train"].features["label"].names`.

## What this project uses

- We hold out the official 3,080-row test set as the **LLM baseline test set** — Claude evaluates all 3,080 queries here
- The 10,003-row train set is the pool we sample training subsets from at `n ∈ {50, 100, 250, 500, 1000, 2500, 5000}` (see `notebooks/01_data_prep.ipynb`)
- Stratification by intent label at each subset size — important because some intents have fewer than 100 training examples

## Class balance notes

Banking77 is reasonably balanced: most intents have 80–200 training examples. A few have fewer, which will bite at small *n*. Plan to:
- Use stratified sampling so every intent appears in every training subset
- Report per-intent F1 + identify intents where the small model systematically fails

## Citation

```bibtex
@inproceedings{Casanueva2020,
  author      = {I{\~{n}}igo Casanueva and Tadas Tem{\v{c}}inas and Daniela Gerz and Matthew Henderson and Ivan Vuli{\'{c}}},
  title       = {Efficient Intent Detection with Dual Sentence Encoders},
  booktitle   = {Proceedings of the 2nd Workshop on NLP for ConvAI - ACL 2020},
  year        = {2020}
}
```
