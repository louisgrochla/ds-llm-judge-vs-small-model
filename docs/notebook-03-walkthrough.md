# Notebook 03 — what it did and why

Plain rundown of the fine-tune sweep. This is where the project's headline result actually gets generated — 40 DistilBERT runs across two recipes, with the v2 retune being the one that crosses Sonnet.

---

## What this notebook produces

- **`results/finetune/v2_retune/n{n}_seed{seed}.parquet`** — 40 per-run prediction files (8 training sizes × 5 random seeds), each with the full 77-class softmax probabilities for every test query
- **`results/finetune_predictions_v2_retune.parquet`** — all 40 runs aggregated into a single parquet (~38 MB)
- **`results/finetune_ensemble_v2_retune.parquet`** — soft-vote ensemble macro-F1 per n (8 rows)
- **`results/checkpoints/v2_retune_n9000_seed0/`** — the model checkpoint we ship to HuggingFace Hub and serve in the demo
- The same set of artefacts in `v1_conservative/` from the earlier sweep, kept for comparison

## Designed to run on either Colab or local

Cell 1 detects whether it's on Colab. On Colab it clones the repo and installs the few packages Colab doesn't ship with; locally it's a no-op. Same notebook works in both environments — no separate Colab-specific version to keep in sync.

The fine-tune sweep needs a GPU to be fast (each run is ~30 sec on T4, ~3 min on Mac MPS). Free Colab T4 finishes the full 40-run v2 sweep in roughly 75 minutes. The notebook saves predictions after each run completes, so a Colab disconnect mid-sweep loses at most one run.

## Recipe versioning

Section 4 introduces a `RECIPE` constant with two values: `'v1_conservative'` and `'v2_retune'`. Each maps to a different hyperparameter set; per-recipe outputs land in separate subfolders (`results/finetune/v1_conservative/`, `results/finetune/v2_retune/`) so both experiments coexist for comparison in notebook 04.

This is the only way to honestly tell the methodology story. The original sweep with conservative defaults — what a typical "I followed the tutorial" fine-tune produces — capped at macro-F1 = 0.77 at n=5,000, well below Sonnet's 0.891. Showing both runs side-by-side in notebook 04 is the diagnostic narrative: *here's what defaults give you, here's what an informed retune gives you, here's why each change mattered.*

## Why v1 didn't work

At LR=2e-5 with batch=32 and 5 epochs at n=5000, you get 5 × ⌈5000/32⌉ = 785 weight updates total. That's enough for the model to learn something, but the std across seeds at n=5000 was 0.009 — extremely tight, meaning the model had **converged** at this LR. More epochs alone wouldn't push it past 0.77. The combination of a too-low LR and not enough updates left the model stuck in a local minimum.

## Why v2 worked

Four targeted changes, each addressing an independent bottleneck:

1. **LR doubled (2e-5 → 4e-5).** Each weight update is bigger, helping the classifier head escape the random-init zone faster. Still below Casanueva 2020's 5e-5 — testing whether a midpoint LR is sufficient before going maximal. (It was.)
2. **Batch halved (32 → 16).** Doubles the number of weight updates per epoch at no compute cost (smaller batches just mean more steps per pass).
3. **Epochs scaled by n.** Small training sets need more passes because each example gets seen more times. `EPOCHS_BY_N = {50: 30, 100: 25, 250: 20, 500: 15, 1000: 12, 2500: 10, 5000: 8, 9000: 8}`. At n=50 with the v1 schedule the model got only 20 × ⌈50/32⌉ ≈ 40 updates — measurably more than 5 epochs but still tight. v2's 30 epochs at smaller batch gives ~120 updates.
4. **Early-stopping patience bumped (2 → 3).** At higher LR the validation loss wobbles before improving — patience 2 was occasionally cutting runs short mid-improvement.

The result: v2 n=100 went from macro-F1 = 0.022 (v1) to 0.15 (v2) — **8× better**. v2 n=5000 went from 0.77 to 0.91, crossing Sonnet. v2 n=9000 ensemble reached 0.934, approximately matching Casanueva 2020's BERT-base full-train number with a smaller model.

## Per-(n, seed) prediction files

Every fine-tune run writes a single parquet file with the test predictions and the full softmax probability distribution per query (3,080 rows × 77-element probability vectors). Three reasons this matters:

**Resumability.** A Colab disconnect mid-sweep doesn't lose completed runs — the skip-if-exists logic in the main loop checks for the parquet's presence before re-training.

**Atomic checkpoints.** One run = one parquet. If the file exists, that (n, seed) is done. No partial state to clean up.

**Softmax saving enables soft-vote ensembling.** With just argmax predictions you can only do hard-vote ensembling (which intent did most seeds predict?). With full softmax distributions you can average probabilities across seeds and argmax the mean — strictly stronger than hard-vote, and the production-realistic comparison since most deployments run multiple model copies.

## Pre-tokenisation as an optimisation

Val and test are constant across all 40 runs — same 500 and 3,080 rows respectively, no shuffling. Section 3 tokenises them once into tensor dictionaries at notebook load time. Each fine-tune run reuses the cached tensors instead of re-tokenising. Training subsets are tokenised inside the loop (they're per-(n, seed) so can't be pre-cached).

Without this, each of the 40 runs would re-tokenise 3,580 val+test rows. Tokenisation is CPU-bound and adds ~5 sec per run; pre-caching saves ~3 minutes total over the sweep. Small win individually, real win at sweep scale.

## Soft-vote ensembling (section 8)

For each training size, stack the 5 seeds' softmax probability tensors (shape `(5, 3080, 77)`), average across the seed axis, take argmax along the class axis. That's the ensemble prediction per test query.

This is the production-realistic headline metric for two reasons:

1. **Single-seed runs vary substantially**, especially at small n. At n=500, std across 5 seeds was 0.032 — about 3 percentage points. Reporting the "best seed" macro-F1 is cherry-picking; reporting the mean ignores that production systems would actually ensemble.
2. **Ensembling consistently adds +0.01 to +0.03** over the single-seed mean across the curve. At n=2,500 it's enough to cross Sonnet's 0.891 (0.884 single-seed → 0.910 ensemble), which is what the project's headline finding is built on.

## Saving the best checkpoint

The main loop has special-case logic for `(n=9000, seed=0)` — it also saves the trained model+tokenizer to `results/checkpoints/v2_retune_n9000_seed0/`. This is the model published to HuggingFace Hub and served in the Gradio demo. We only need one canonical model; n=9,000 seed=0 is the simplest defensible choice (largest training set, first seed, fully reproducible).

Edge case: if the predictions for n=9000, seed=0 already exist on disk but the checkpoint doesn't (e.g., kernel killed between predictions save and model save), the loop retrains that one run specifically to capture the checkpoint. This catches the "you have predictions but no deployable model" failure mode that would otherwise require a fresh sweep.

## What the section 9 summary table tells you

For each n, the table reports:
- Per-seed macro-F1 (5 numbers showing variance)
- Mean ± std across seeds (the central tendency and reliability)
- Ensemble macro-F1 (the production-realistic headline)

Three things to read off it:

1. **Standard deviation should collapse at large n.** At n=9,000 v2 we see std = 0.003 — essentially zero. That confirms convergence; the model isn't just lucky at high n, the result is stable.
2. **Ensemble should beat the seed-mean by a small but consistent margin.** At n=2,500 the seed-mean is 0.884 and the ensemble is 0.910 (+0.026). At n=9,000 the seed-mean is 0.925 and the ensemble is 0.934 (+0.009). The ensemble boost shrinks as variance shrinks, but stays positive.
3. **The crossover with Sonnet (0.891) is where the ensemble line meets the horizontal baseline.** For v2 that's between n=1,000 (0.866) and n=2,500 (0.910) — the crossover is at n=2,500 in the discrete grid we tested. For tighter resolution you'd interpolate (which the calculator page does).

## Summary

Notebook 03 is the project's most compute-heavy notebook and the source of every quality number in the README, calculator, and notebook 04 analysis. Its single most important design choice — recipe versioning + soft-vote ensembling + per-(n, seed) atomic files — is what makes the v1→v2 retune story defensible: both experiments live side-by-side on disk, the comparison is reproducible, and the ensemble metric is the honest production number rather than a cherry-picked best seed.
