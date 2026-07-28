# Build-vs-buy decision table

> **Correction, 28 July 2026.** This table previously carried a "Meets Sonnet?" column and gated
> its recommendations on `macro-F1 ≥ 0.8913`. That baseline is withdrawn: the LLM's test set was
> collected ordered by class, so each prompt implicitly narrowed the choice from 77 intents to
> about four, and 0.8913 is not a measurement of zero-shot performance. The quality-parity column
> and every recommendation derived from it have been removed. See the
> [correction notice](../README.md#correction-notice--28-july-2026).
>
> **The cost columns are unchanged** — they are arithmetic on published list prices.

Cost comparison for Banking77-style intent classification: self-hosted DistilBERT against a
frontier LLM API.

## Assumptions

| Input | Value | Provenance |
|---|---|---|
| API per-query cost | $0.0053 uncached / $0.0015 cached | Anthropic list price ($3/$15 per MTok), 2026-05-24, applied to an **estimated** ~1,500 input + ~50 output tokens |
| Self-host fixed cost | $360/month | HF Inference Endpoint T4 at $0.50/hr × 24 × 30 |
| Self-host per-query cost | $0.000007 | Assumes 200 inferences/sec at batch=8 and **10% utilisation**. No timing artifact for either exists in this repo. |
| Labeling cost | $0.50/example | Internal labeller / crowd floor, conservative |

Break-even volume = fixed monthly cost ÷ API per-query cost: **~68,600 queries/month** uncached,
**~245,000** with prompt caching.

## Table

Quality is the 5-seed soft-vote ensemble from the fine-tuning sweep. Note the mismatch this table
cannot resolve for you: quality is an **ensemble** figure while the self-host cost prices a
**single** model.

| Scenario | n labels | queries/mo | Ensemble macro-F1 | API $/mo | Self-host $/mo | Cheaper option | Labels pay back in |
|---|---|---|---|---|---|---|---|
| Hobby / prototype | 100 | 1,000 | 0.384 | $5 | $360 | API | — (API is cheaper) |
| Small B2B SaaS | 1,000 | 50,000 | 0.866 | $263 | $360 | API | — (API is cheaper) |
| Mid-market fintech | 2,500 | 500,000 | 0.910 | $2,625 | $360 | Self-host, saves $2,265/mo | 0.6 months |
| Scaled product (Monzo/Revolut tier) | 5,000 | 5,000,000 | 0.927 | $26,250 | $360 | Self-host, saves $25,890/mo | 0.1 months |
| Public-cloud SaaS | 9,000 | 50,000,000 | 0.934 | $262,500 | $360 | Self-host, saves $262,140/mo | under a week |

"Cheaper option" is a cost verdict only. Whether the in-house model is **good enough** depends on
a quality bar this repo cannot currently supply — the quality column tells you what each label
budget bought on Banking77, not whether it beats an LLM.

## A caveat worth more than the table

A character n-gram TF-IDF model with logistic regression — no transformer, no GPU, trains in
seconds on a laptop — reaches macro-F1 0.8505 (single seed) / 0.8873 (5-seed ensemble) at
n=2,500, and 0.9007 / 0.9014 at n=9,000 on this same task. That is within about 3 points of the
fine-tuned DistilBERT at every budget above n=1,000, for a rounding error of the compute.

Before budgeting for a fine-tuned transformer *or* an LLM, measure that baseline on your own
data. Reproduce with `python scripts/run_frozen_encoder.py --encoder tfidf`.
