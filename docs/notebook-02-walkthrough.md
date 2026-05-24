# Notebook 02 — what it did and why

Plain rundown of what notebook 02 does, the reasoning behind each design choice, and the actual results.

---

## What this notebook produces

The LLM-as-judge baseline that notebook 03's fine-tuned DistilBERT is trying to match. Specifically: predictions from Claude Sonnet 4.6 on Banking77's official 3,080-row test set, saved to `results/llm_predictions_v1_test.parquet` for notebook 04's bootstrap significance test.

**Headline result for v1 on the full test set:**
- Accuracy: 0.8981
- Macro F1: 0.8913
- Weighted F1: 0.8913
- Hallucinations: 0

## Why Claude Code instead of the Anthropic API

The two serve the same Sonnet 4.6 weights, but billing is separate. Claude Code runs against my existing Claude Max subscription (project cost: $0). The API would have cost ~$25 across iterations. Tradeoff: I lose per-call latency measurements and have to paste prompts manually instead of looping. For anyone reproducing this with API access, `src/prompts/v1.txt` is the same prompt — running it via the Anthropic SDK should produce near-identical results.

## The batched paste-and-parse architecture

Each "batch" is a paste-ready prompt containing the full 77-intent label list, 150 numbered queries, and instructions to return a single JSON array of `{id, intent}` objects. One paste, one response, 150 predictions per cycle.

Why batches of 150:
- Output stays comfortably inside Claude's default output token limit (~3,750 tokens for 150 JSON objects vs ~4,096 default cap)
- Trade-off between paste fatigue (smaller batches = more pastes) and output reliability (larger batches risk truncation)
- 150 felt empirically right; 300 might also work with Sonnet's higher output ceiling, but no point pushing limits unnecessarily

The full dev_slice (300 rows) became 2 batches. The full test set (3,080 rows) became 21 batches.

## Prompt versioning

`src/prompts/v1.txt` is a template with placeholders for `{label_list_block}`, `{query_block}`, `{start_id}`, `{end_id}`. The notebook fills these per batch via `src/llm_eval.build_batch_prompt()`. The template captures the *wording* of instructions (which is what changes between prompt versions); the label list and query block are mechanically constructed.

If I'd needed to iterate to a v2, the workflow would have been: copy `v1.txt` to `v2.txt`, edit the wording, change `PROMPT_VERSION = 'v2'` at the top of the notebook, re-run. Diffable text files, no notebook editing.

In practice, v1 was good enough so v2 never happened.

## Dev iteration strategy (why we didn't burn the test set)

Banking77's official 3,080-row test set is the headline number — running anything on it should happen once, after the prompt has converged. Otherwise you're optimizing the prompt against the same data you'll later report metrics on, which inflates results.

The 300-row dev slice (carved from the train pool in notebook 01, disjoint from test) is for iteration. Each iteration costs ~5 minutes of paste-and-parse work, vs ~30-40 minutes for the full test. v1 was tested on dev first; results were credible (0.8076 macro-F1 with all the "wrong" answers being defensible Banking77 ambiguities), so v1 went straight to test.

## Why I didn't iterate to v2

After scoring v1 on dev, the wrong predictions broke down roughly as:
- ~40% genuine Banking77 label ambiguity (Sonnet picked a defensible alternative — e.g. `change_pin` for "I haven't received my PIN yet" instead of `get_physical_card`)
- ~30% real model confusion between similar intents (could potentially improve with prompt tweaks)
- ~30% clear model errors

Even a perfectly-tuned v2 might recover 30% of the wrong predictions — bumping accuracy from 81.7% to ~87% on dev. Real improvement but limited; the rest is dataset-level noise we can't fix. Time spent on v2 is time not spent on notebook 03 (the fine-tune sweep, which is the actual differentiator of the project). Decision: ship v1, document the failure modes, move on.

## The dev → test gap (0.8076 vs 0.8913 macro-F1)

The 8-point jump from dev to test isn't suspicious — it's a known property of macro-F1 at small per-class samples. Dev has ~4 examples per intent (300/77); test has ~40 (3080/77). A few unlucky errors on rare classes at small n drag per-class F1 down sharply, and macro-averaging amplifies it. At test scale, per-class F1 stabilizes.

Two things rule out contamination:
- v1 prompt was identical between dev and test runs (no modification)
- Test set was never shown to Sonnet during dev iteration (dev was carved from train pool)

The 0.8913 test number is the trustworthy headline. The 0.8076 dev number is a noisy point estimate that mostly serves to confirm "the prompt isn't broken."

## Zero hallucinations across 3,380 predictions

Combined across dev (300) + test (3,080), Sonnet produced 3,380 predictions and 0 used an intent name outside the 77-class list. This validates the prompt design — the explicit "exact intent name from the list above" instruction plus the literal label list in the prompt is enough to prevent invention. No prompt engineering needed beyond this.

## Top confusion patterns at test scale

The most frequent wrong predictions:
- `get_physical_card` ↔ `change_pin` (40 confusions): PIN-related queries that Banking77 labels as physical-card delivery issues. Counterintuitive labeling; Sonnet's `change_pin` picks are arguably more accurate.
- `order_physical_card` ↔ `get_physical_card` (36): the get/order verb ambiguity that surfaced in the original sanity test, scaled up.
- `beneficiary_not_allowed` ↔ `declined_transfer` (14): generic "I tried to transfer and it failed" queries getting the broader `declined_transfer` label.
- Various `_charge`/`_fee` pair confusions where the query doesn't specify card vs bank transfer.

These are mostly Banking77 label noise. The same patterns will likely show up in DistilBERT's failure modes (notebook 03), and notebook 04's failure-mode analysis will use this overlap to argue that the gap between LLM and small model is partly a dataset ceiling, not a pure model-capability story.

## What notebook 04 consumes

- `results/llm_predictions_v1_test.parquet` — per-prediction dataframe with `pred_id`, `pred_intent`, `true_intent`, `text`, `is_valid_intent`, `is_correct`. Notebook 04 joins this against the DistilBERT predictions parquet from notebook 03 on `pred_id` to do the paired bootstrap test on the macro-F1 difference at the candidate crossover *n*.
- `results/llm_summary_v1_test.json` — the headline numbers + top 10 confusion pairs, for quick comparison and the final writeup.

## Reproducibility

Everything needed to reproduce this notebook's results (without running Claude or paying for API calls) is in the repo:
- `src/prompts/v1.txt` — the exact prompt template
- `results/eval_batches/v1/test/batch_*.txt` — the 21 paste-ready prompts that went into Claude Code
- `results/eval_responses/v1/test/batch_*.txt` — the 21 raw JSON responses Sonnet returned
- The notebook's parse + score logic — deterministic given the responses

A reader can verify by reading any batch prompt against the matching response, or by re-running sections 3-6 of the notebook against the committed responses (regenerates `llm_predictions_v1_test.parquet`).
