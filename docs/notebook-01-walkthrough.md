# Notebook 01 — Walkthrough

For each section: **What** the code does, **Why** this design choice (including rejected alternatives), and **How to verify** for yourself — both concrete commands and methodology challenges that push you to defend the choice.

> Read this with `notebooks/01_data_prep.ipynb` open beside it. The walkthrough doesn't reproduce the code, only the reasoning.

---

## Section 1 — Load Banking77

**What it does.** Downloads the Banking77 dataset (10,003 train + 3,080 test queries, 77 intents) from HuggingFace Hub. Two API calls: one for each parquet file (train + test), one for `dataset_infos.json` to get the 77 intent names. Everything lands in pandas dataframes.

**Why this approach.** The "obvious" call — `load_dataset('PolyAI/banking77')` — triggers HuggingFace's old script-based loader. That script tries to fetch CSVs from a GitHub URL that no longer exists, so it fails. Bypassing it by loading the parquet files directly from HF Hub's auto-conversion (`refs/convert/parquet` revision) is more reliable and doesn't depend on a deprecated code path.

**Rejected alternatives.**
- Downgrading the `datasets` library to v2 (some people pin it). Works, but you're depending on a deprecated API. If `datasets` drops v2 support in 6 months, the notebook breaks again.
- Hosting our own copy of Banking77. Adds a maintenance surface for no benefit.

**Verify it yourself.**
```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('data/processed/test.parquet')
print(df.shape)       # (3080, 2)
print(df.columns)     # ['text', 'label']
print(df.head())
"
```

**Methodology challenge.** Banking77 is *one* English-language banking dataset. The crossover finding you publish — "DistilBERT matches Sonnet at n=X examples" — is for THIS task. Would the same crossover hold on CLINC150 (broader intent set)? On a multilingual customer-service dataset? You don't know; "Limitations" in the README should call this out. Defending the dataset choice means knowing what it does and doesn't represent.

---

## Section 2 — Schema audit + class balance

**What it does.** Checks that there are no null `text` or `label` values. Confirms dtypes. Computes per-intent example counts in the train set, prints summary stats and the rare-intent list, plots a histogram.

**Why this matters.** Two real risks:
1. **Nulls in labelled data.** A single null label that silently survives into training will throw a confusing tokenizer error or crash a metric. Better to catch upfront.
2. **Class imbalance.** Rare classes get squeezed out in small-n training subsets. Banking77's range is 35–187 examples per intent — only two intents under 50 (`contactless_not_working` at 35, `virtual_card_not_working` at 41). "Reasonably balanced" means our experiment design survives.

**Verify it yourself.**
```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('data/processed/test.parquet')
print('nulls:', df.isnull().sum().to_dict())
print('class counts (head):')
print(df.label.value_counts().head())
"
```

**Methodology challenge.** What if Banking77 had 5 intents with only 5 examples each? Even our 5-seed strategy at n=50 couldn't pull a usable subset of those rare classes — every model would systematically fail on them. Would you still claim "DistilBERT matches Sonnet at n=X"? The honest answer is "matches on the well-represented classes; fails on the rare tail." This is why the analysis section breaks out per-class F1, not just macro.

---

## Section 3 — Token length distribution

**What it does.** Runs the DistilBERT tokenizer over every training query, records the resulting token counts (including the `[CLS]` and `[SEP]` special tokens), plots a histogram, prints percentiles and the percentage of queries that would be truncated at MAX_LENGTH ∈ {32, 64, 128}.

**Why this matters.** `MAX_LENGTH` is a knob you set in notebook 03 — it caps how many tokens DistilBERT processes per query. Higher = more GPU memory + slower training. Lower = faster training but you start losing content. The right value is the smallest that doesn't truncate meaningful information.

**Why measure tokens, not words.** BERT tokenizers split unfamiliar words into subwords (e.g. "withdrew" → "withdr" + "##ew" = 2 tokens). Counting whitespace-separated words undercounts by 20–40% on natural text. Measuring with the actual tokenizer gives the real number.

**Result.** 99th percentile = 53 tokens. `MAX_LENGTH=64` truncates 0.30% of queries (≈30 out of 10,003) — negligible, with comfortable headroom.

**Verify it yourself.**
```bash
.venv/bin/python -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('distilbert-base-uncased')
print(len(t.encode('I noticed an extra fee when I withdrew money.')))  # ~14
print(len(t.encode('Hello.')))  # 4 — [CLS] hello . [SEP]
"
```

**Methodology challenge.** What would happen at `MAX_LENGTH=16`? Section 3 shows you'd truncate ~50% of queries — the model would never see the back half of most inputs. Banking queries often *end* with the operative phrase ("…before my card expires?"). Truncating at 16 would make the model worse on those queries for reasons that have nothing to do with model capacity. That's why you set `MAX_LENGTH` empirically, not by guess.

---

## Section 4 — Test holdout confirmation

**What it does.** Compares the class distribution in train vs the official test split. Computes KL divergence, plots side-by-side bars.

**Why this matters.** If the test set over-represents some intents and under-represents others compared to train, the "macro-F1 on test" number you report wouldn't reflect what you'd get on a fresh sample from the same distribution. Banking77's curators built the splits to match, but you confirm rather than trust.

**Result.** KL(test ‖ train) = 0.040 — distributions match closely. All 77 classes present in test (important: missing classes have undefined metrics, which would silently corrupt your macro-F1).

**Verify it yourself.**
```bash
.venv/bin/python -c "
import pandas as pd
from scipy.stats import entropy
train = pd.read_parquet('data/processed/test.parquet').label.value_counts(normalize=True).sort_index()
# (using test as proxy here; in real check, reload both)
print(f'Classes present: {len(train)}')
"
```

**Methodology challenge.** What if KL were 0.5 instead of 0.04? Your "macro-F1 = 85% on test" claim would not transfer to a production deployment, because production distribution would be train-like (most-common intents first), but the eval was over-weighting rare ones. You'd be reporting an artificially low number. Dataset distribution mismatch is a *very* common source of model "regressions" that aren't actually regressions.

---

## Section 5 — Carve fixed validation set

**What it does.** Pulls 500 stratified rows out of the 10,003 train pool. Same 500 rows used as the val set for every (n, seed) fine-tune in notebook 03.

**Why this matters.** During fine-tuning you need a holdout to do early stopping — stop training when val loss stops improving, otherwise you overfit. The val set has to be separate from train (no leakage) and constant across runs (controlled experiment).

**Why stratified.** Random 500-row sample could under-represent rare intents. With 77 intents and only 35–41 examples for the rarest two, a non-stratified sample of 500 might pull only 1–2 rare-intent examples. Stratified guarantees ≈6.5 per intent on average — enough to detect regression on every class.

**Why 500 specifically.** Big enough for stable per-class metrics; small enough not to eat the training budget (10,003 → 9,503 remaining is barely affected).

**Why constant across n.** If val differed per (n, seed), early-stopping decisions wouldn't be comparable across runs. Fixed val = clean ablation.

**Verify it yourself.**
```bash
.venv/bin/python -c "
import pandas as pd
val = pd.read_parquet('data/processed/val.parquet')
print('size:', len(val))
print('classes:', val.label.nunique())
print('per-class min/max:', val.label.value_counts().min(), '/', val.label.value_counts().max())
"
```

**Methodology challenge.** What if you used a 50-row val set instead of 500? With 77 classes, some classes would have 0 val examples — you'd be unable to early-stop on those classes' performance. Worse, the val loss would be dominated by whatever 50 examples happened to be sampled. Tiny val sets make every training run high-variance and uncomparable.

---

## Section 6 — Carve dev slice for LLM prompt iteration

**What it does.** Pulls 300 stratified rows from what remains of the train pool (after val carve). Saved as `dev_slice.parquet`. Used in notebook 02 for iterating the LLM prompt without touching test.

**Why this matters.** Running Claude Sonnet 4.6 on the full 3,080-row test set costs ~$15 per pass. You'll iterate the prompt 5–8 times (try a new instruction, see if metrics improve, refine). At $15 × 8 = $120, that's wasteful. A 300-row dev slice costs ~$1.50 per iteration → ~$12 across 8 iterations + ~$15 for ONE final test run = ~$27 total instead of $120.

**Why the dev slice has to come from the train pool, not test.** This is the most important principle in ML evaluation. If you iterate prompts on a slice of test, you're picking the prompt that maximises score on those specific test examples. Your final "macro-F1 = 85% on test" would be optimistic — anyone reproducing on a fresh sample of similar data would get a lower number. The technical name is "test set contamination" or "evaluation leakage." It silently inflates almost every public ML benchmark result; the fix is to never look at test until the very end.

**Verify it yourself.**
```bash
.venv/bin/python -c "
import pandas as pd
dev = pd.read_parquet('data/processed/dev_slice.parquet')
test = pd.read_parquet('data/processed/test.parquet')
overlap = set(dev.text) & set(test.text)
print(f'dev size: {len(dev)}')
print(f'classes in dev: {dev.label.nunique()}')
print(f'overlap with test (must be 0): {len(overlap)}')
"
```

**Methodology challenge.** What if you iterated the prompt directly on test, picked the best, then reported test metrics? Your number would look great. But you'd be conflating "what the prompt achieves on data the prompt was tuned on" with "what the prompt achieves on data the prompt has never seen." A reproducible study makes the second claim. The first claim is, technically, fraud — even if unintentional.

---

## Section 7 — Training subsets across seeds (the central methodology call)

**What it does.** For each of 5 random seeds, builds an interleave-by-class shuffle of the 9,203 remaining train pool, then takes the first n rows for n ∈ {50, 100, 250, 500, 1000, 2500, 5000}. That's 7 × 5 = 35 parquet files in `data/processed/train_n{n}_seed{seed}.parquet`.

**Why multiple runs per training size.** Fine-tuning is noisy. A single fine-tune at n=500 might land at macro-F1 = 0.74 or 0.81 depending purely on which 500 examples happened to be in the subset, plus the random weight initialisation of the classifier head. To make a defensible claim like "DistilBERT matches Sonnet at n≈X," you need to know the *variance* across runs, not just one point estimate. Standard practice: report mean ± 95% CI across multiple runs.

**Why random seeds, not k-fold cross-validation.** This is the most important decision in the notebook, so let's unpack it.

K-fold splits your training data into k chunks, trains on k-1, evaluates on the held-out chunk, rotates. With 77 classes and n=50, 5-fold gives you 5 batches of 10 training examples each — and at any rotation, the training set has 40 examples spread across 77 classes. **Most classes will have zero training examples per fold.** The model can never predict those classes correctly. The resulting macro-F1 number for the n=50 fold isn't measuring "what does DistilBERT learn at n=50" — it's measuring "what's the macro-F1 of a model that can't even attempt 40+ of the 77 classes." Useless.

Random seeds avoid this. Each seed gives a full draw of n examples from the pool. At n=50 some classes are still missing (you can't fit 77 classes into 50 examples), but at n=100 every class is present at every seed (confirmed by the coverage table in the notebook output). The variance estimate is just as good as k-fold's.

**Why the "round-robin across labels" shuffle.** A pure random shuffle could, at n=50, pull 50 examples from just 10 lucky classes. Round-robin: take one example from each class (shuffled within class), cycle through all classes, repeat until you've drawn enough. This spreads examples maximally across classes at small n. At larger n it doesn't matter (you exhaust some classes and the shuffle becomes effectively random) — but at small n it's the difference between "50/77 classes present" and "10/77."

**Why nested subsets.** For seed=0, the n=100 subset is constructed to include all 50 rows of seed=0's n=50 subset plus 50 more. This means comparing performance across n at the same seed isolates "what happens when I add more data" rather than confounding it with "what happens when I sample different data." Cleaner experimental design, easier to reason about results.

**Verify it yourself.**
```bash
.venv/bin/python -c "
import pandas as pd
n50_s0 = pd.read_parquet('data/processed/train_n50_seed0.parquet')
n100_s0 = pd.read_parquet('data/processed/train_n100_seed0.parquet')

print(f'n=50, seed=0: {len(n50_s0)} rows, {n50_s0.label.nunique()} classes')
print(f'n=100, seed=0: {len(n100_s0)} rows, {n100_s0.label.nunique()} classes')

# Nesting check — every row in n=50 should appear in n=100
is_nested = set(n50_s0.text).issubset(set(n100_s0.text))
print(f'nested (must be True): {is_nested}')
"
```

**Methodology challenge.** What if you used pure random shuffle (not round-robin)? At n=50 you might pull 50 examples from 10 classes, leaving 67 unrepresented. Your "n=50 macro-F1" would be conditioned on a freak sample, and different seeds would land in wildly different parts of the class space — high variance for the wrong reason (sampling, not training). Round-robin reduces "sampling variance" so the variance you measure across seeds is the variance you care about: model initialisation + training stochasticity.

---

## Section 8 — Save processed splits + label names

**What it does.** Writes `test.parquet` (3,080 rows) and `label_names.txt` (77 lines, one intent per line, position = label ID). Together with the val / dev / per-(n, seed) files from sections 5–7, this is the complete on-disk contract that notebooks 02–04 depend on.

**Why parquet for data, plain text for labels.**
- **Parquet for dataframes** preserves dtypes (label stays `int64`, not `object`), is faster to read than CSV at scale, and uses less disk.
- **Plain text for labels** because the list is tiny (1.6 KB), human-readable, easy to inspect with `cat`, and doesn't need a deserialiser. Position in the file = label ID — implicit but obvious.

**Why this is the contract.** `src/data.py` exposes `load_test_set()`, `load_val_set()`, `load_train_subset(n, seed)`, `load_label_names()`. Every other notebook calls these — they don't know about parquet, the file paths, or the directory layout. If you ever change the file layout, you update `src/data.py` once and every notebook keeps working.

**Verify it yourself.**
```bash
wc -l ~/Desktop/ds-llm-judge-vs-small-model/data/processed/label_names.txt   # 77
head -5 ~/Desktop/ds-llm-judge-vs-small-model/data/processed/label_names.txt  # activate_my_card, age_limit, ...

# And via the helper:
cd ~/Desktop/ds-llm-judge-vs-small-model && .venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from src.data import load_test_set, load_label_names
test = load_test_set()
labels = load_label_names()
print(f'test shape: {test.shape}')
print(f'first 5 labels: {labels[:5]}')
print(f'intent for row 0: {labels[test.iloc[0][\"label\"]]} — query: {test.iloc[0][\"text\"]}')
"
```

**Methodology challenge.** What if `src/data.py` didn't exist and notebooks 02–04 read parquet paths directly? Two failure modes:
1. If you ever change the file layout (e.g., rename `train_n{n}_seed{seed}.parquet` to `train/{n}/seed-{seed}.parquet`), you'd have to edit every notebook. The helper-module pattern is what lets you refactor without spelunking.
2. Multiple notebooks loading the same data with different conventions (some treating label as `int`, some as `str`) would produce subtle metric bugs. Central loading enforces consistency.

This is "DRY" applied to data plumbing, not just code. Pays off the first time you change the file layout.

---

## How this walkthrough fits the project

Notebook 01 produces the data that every other notebook consumes. If you understand why each split exists and what it guarantees, the rest of the project is much easier to reason about:
- The dev slice exists because LLM iteration is expensive (section 6) → notebook 02 knows where to iterate cheaply
- The val set is fixed because comparisons across n must control for early stopping (section 5) → notebook 03 uses the same val every run
- Training subsets are nested within seed (section 7) → notebook 04 can plot "performance vs n at fixed seed" with confidence
- The 5-seeds choice instead of k-fold (section 7) → notebook 04 has a clean variance estimate to put on its figures

Every choice in 01 was made *for* a downstream notebook. If something feels arbitrary, look at how the downstream notebook uses it — the constraint usually comes from there.
