# Notebook 01 — what it did and why

Plain rundown of what each section does and the reasoning behind it. Reference doc — open this if you come back to the project months later and need to remember why a choice was made.

---

## 1. Load Banking77

Downloads the dataset from HuggingFace Hub — 10,003 train and 3,080 test queries across 77 banking intents. The standard `load_dataset('PolyAI/banking77')` call doesn't work: Banking77's loader script tries to fetch CSV from a GitHub URL that was removed. Instead, I load the parquet files directly from HF Hub's auto-conversion (every dataset on HF gets parquet exports generated automatically). Same data, no broken-script dependency.

## 2. Schema audit + class balance

Checked for nulls (none in either column), confirmed dtypes, computed per-class counts. Most intents have 100–180 training examples; two outliers sit under 50: `contactless_not_working` (35) and `virtual_card_not_working` (41). These two will be hardest to learn — at our smallest training subsets (n=50, n=100) they may have zero or one example, and even at moderate n the model has less to learn from than for typical intents.

## 3. Token length distribution

Ran the DistilBERT tokenizer over every query to count actual tokens. Important to use the real tokenizer rather than whitespace word counts — BERT splits unfamiliar words into subwords (e.g. "withdrew" becomes 2 tokens), so word counts undercount by 20–40% on natural text. Result: median 13 tokens, 99th percentile 53. This confirms `MAX_LENGTH=64` in the fine-tune notebook is the right setting: it covers 99.7% of queries with no meaningful GPU memory cost.

## 4. Test holdout confirmation

Compared train vs test class distributions using KL divergence. Got 0.04, very close to 0, meaning the test set looks like a representative sample of train. If they didn't match (KL of 0.5+), the macro-F1 numbers we report on test wouldn't predict how the model behaves on fresh data — a common source of phantom regressions in production.

## 5. Val carve (500 stratified rows)

Pulled 500 stratified rows from train as a fixed validation set. Used during fine-tuning for early stopping (stop training when val loss plateaus). The same 500 rows are used for every (n, seed) combination in notebook 03. Fixed across runs because comparing models trained at different n requires the early-stopping decisions to be controlled — variable val per run would add noise that masquerades as signal. 500 rows is big enough that every one of the 77 classes is represented (~6.5 per class on average), small enough not to dent the training pool.

## 6. Dev slice (300 stratified rows from train pool)

The LLM iteration playground for notebook 02. Running Claude Sonnet 4.6 on the full 3,080-row test set costs ~$15 per pass; iterating prompts that way would burn $100+ in budget. The 300-row dev slice costs ~$1.50 per pass, so prompt iteration is affordable. Then one final run on the full test set for the headline number.

Carved from the train pool, **not** from the test set, for one critical reason: if you iterate prompts on test data, you're selecting whichever prompt happens to maximise score on those specific examples — and then reporting that score as your "test result." That number is inflated. Anyone running the same prompt on a fresh sample would get lower numbers. The dev slice keeps prompt selection honest.

## 7. Training subsets across seeds

Generated 35 parquet files — 7 training sizes (50, 100, 250, 500, 1000, 2500, 5000) × 5 random seeds. Each subset within a seed is nested (the n=100 subset for seed=0 contains all 50 rows of the n=50 subset for seed=0), so comparing across n at fixed seed isolates "what happens when I add more data" from "what happens when I sample different data."

The key decision: used **random seeds instead of k-fold cross-validation**. At n=50 with 77 classes, k-fold splits 50 examples into 5 batches of 10. The model trains on 40 examples spread across 77 classes — most classes get zero training examples per fold, so the model has no possible way to predict them, and the macro-F1 number becomes meaningless. Random seeds avoid this: each seed is a full independent draw of n examples. At n=50 some classes are still missing (you can't fit 77 classes into 50 examples), but at n=100+ every class is present at every seed (confirmed in the notebook output's coverage table). Same variance estimate as k-fold would give, no degenerate-fold problem.

Used a round-robin-across-labels shuffle instead of pure random. With 77 classes, a pure random sample of 50 might pull from only 10 classes; round-robin guarantees you spread examples across classes as evenly as possible. At larger n the effect washes out, but at small n it's the difference between a tractable experiment and a meaningless one.

## 8. Save processed splits

Wrote `test.parquet` (3,080 rows) and `label_names.txt` (77 intents, position in file = label ID). Together with the val / dev / 35 training files from sections 5–7, this is the complete contract that notebooks 02–04 depend on.

Parquet for dataframes because it preserves dtypes (label stays as int64, not string), is smaller on disk, and reads faster than CSV. Plain text for label names because the list is tiny (1.6 KB), human-readable, and doesn't need a deserialiser.

The `src/data.py` helper exposes `load_test_set()`, `load_val_set()`, `load_train_subset(n, seed)`, `load_label_names()`. Every other notebook calls these helpers — they never touch file paths directly. If I ever change the file layout (e.g., add a new directory, rename a file), I update `src/data.py` once and every notebook keeps working.

---

## Summary

Notebook 01 produces the data that every other notebook consumes. The val set exists for notebook 03's early stopping. The dev slice exists for notebook 02's prompt iteration. The 35 training subsets exist for notebook 03's fine-tune sweep. The test set exists for the one-time final eval in both 02 (LLM) and 03 (DistilBERT). Every choice in 01 was made with a specific downstream consumer in mind.
