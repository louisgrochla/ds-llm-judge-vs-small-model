# Notebook 04 — what it did and why

Plain rundown of the analysis notebook — the one that turns 40 fine-tune runs + the LLM baseline into the publishable artefacts: the crossover plot, the significance test, the cost analysis, and the failure-mode breakdown.

---

## What this notebook produces

Five concrete files that the README, the calculator page, and the LinkedIn writeup all reference:

- **`results/figures/crossover.png`** — the headline chart (macro-F1 vs n with Sonnet baseline + crossover annotation)
- **`results/figures/cost_breakeven.png`** — API monthly cost vs self-host monthly cost by query volume
- **`results/figures/failure_overlap.png`** — stacked bar showing where the two models agree/disagree
- **`results/analysis_summary.json`** — structured JSON with every number quoted in the README
- **`docs/cost-decision-table.md`** — practitioner-facing build-vs-buy table by scenario

Everything downstream (README embeds, calculator's quality curve, LinkedIn post numbers) consumes from those five files. Notebook 04 is the single source of truth for the project's headline numbers.

## Section 1 — load all predictions

Pulls four parquets into memory:

- `llm_predictions_v1_test.parquet` (Sonnet 4.6 on the 3,080-row test set, from notebook 02)
- `finetune_predictions_v2_retune.parquet` (all 40 v2 runs, aggregated)
- `finetune_ensemble_v2_retune.parquet` (per-n soft-vote ensemble macro-F1)
- `finetune_predictions_v1_conservative.parquet` (the original sweep — used as the "before" line on the crossover chart)

Loading v1 alongside v2 isn't just for completeness — the crossover chart explicitly shows both, because the v1→v2 retune story is the methodology punchline. "Here's what conservative defaults give you. Here's what an informed retune gives you." Without the v1 line, the chart is just "fine-tune beats LLM" — uninteresting compared to many published results. With it, the chart tells the diagnostic story.

## Section 2 — headline crossover plot

X axis: training set size on a log scale (n=50 → n=9,000 spans more than two orders of magnitude). Y axis: macro-F1.

Three lines:
- **v1 conservative** (light grey, dotted, x markers) — the "before" reference. Shows what running the textbook defaults gets you.
- **v2 single-seed mean with 95% CI band** (steel blue, circle markers, shaded band) — what a single fine-tune run achieves at each n. The band visualises variance across seeds.
- **v2 ensemble** (dark blue, solid, square markers, thick line) — the production-realistic number. The deliberately bolder styling signals this is what readers should focus on.

Horizontal red dashed line: Sonnet 4.6 baseline at macro-F1 = 0.8913.

The crossover annotation points at the smallest n where the ensemble line crosses Sonnet — found programmatically by iterating `ens_df` in n-order and stopping at the first row where `ensemble_macro_f1 >= LLM_MACRO_F1`. For v2 that's n=2,500. The annotation includes the exact value (0.910) so readers don't have to squint at the y-axis.

Two design choices worth knowing:
- **Log-scale x-axis** because n spans 50→9,000. Linear would compress everything below n=1,000 into the left edge.
- **CI band uses 1.96×std/√5** — standard 95% CI of the mean across 5 seeds, not 95% CI of an individual seed. The band shows "where the mean is likely to land" not "where any single run might land." This matches what the ensemble line represents.

## Section 3 — paired bootstrap test

> **Withdrawn 2026-07-28.** Everything in this section is a difference against the LLM baseline,
> which is invalid — the test set was collected ordered by class. Two further corrections that
> apply regardless: this computes **confidence intervals, not a p-value** (the README's
> "*p* < 0.01" was never computed anywhere), and it resamples **test rows only**, holding the 5
> training seeds fixed, which understates uncertainty. `src/stats.py` now provides a real paired
> permutation test, a bootstrap over seeds *and* rows, and an interval on the crossover budget
> itself; `scripts/reanalysis.py` runs them.

The means-look-close-enough question: is the +0.019 ensemble vs Sonnet gap at n=2,500 statistically real, or could it plausibly be sampling noise?

Implementation: for each n, load all 5 seeds' softmax probabilities, average them per test row, take argmax → ensemble predictions. Then call `src.eval.paired_bootstrap_macro_f1(y_true, ensemble_preds, sonnet_preds, n_bootstrap=2000)` which:
1. Resamples test instances with replacement, same indices applied to both prediction arrays (preserves the pairing)
2. Recomputes the macro-F1 difference on each bootstrap sample
3. Returns the observed gap + 2.5th / 97.5th percentile of the bootstrap distribution as a 95% CI

The output table shows the gap and CI at each n. For v2:
- n=1000: gap = -0.026, CI = [-0.039, -0.013] → Sonnet wins, significantly
- n=2,500: gap = +0.019, CI = [+0.006, +0.030] → **DistilBERT wins, significantly** (CI cleanly excludes zero)
- n=5,000: gap = +0.036, CI = [+0.025, +0.047] → DistilBERT wins decisively
- n=9,000: gap = +0.043, CI = [+0.032, +0.054] → DistilBERT wins decisively

Why paired bootstrap rather than McNemar's test or just reporting the point estimates: paired bootstrap works on any metric (McNemar's only works for accuracy-style binary correctness, not macro-F1). Pairing matters because Sonnet and DistilBERT are evaluated on identical test queries — if you resample independently you throw away that correlation and get artificially wide CIs.

## Section 4 — cost analysis per inference

Hard pricing constants at the top of the cell:
- Sonnet 4.6 API: $3/MTok input, $15/MTok output, $0.30/MTok cached read
- Hugging Face Inference Endpoint T4: $0.50/hour

Two cost functions:

`sonnet_cost_per_query(cached)` — either treats the whole 1,500-token prompt as input ($0.0053/query) or treats only the 100-token query as fresh input + the 1,400-token label list as a cached read ($0.0015/query, ~70% cheaper).

`distilbert_cost_per_query(utilisation)` — fixed `$0.50/hour ÷ (200 inferences/sec × 3600 sec)` divided by utilisation fraction. Two scenarios reported: 10% utilisation (realistic — intermittent load = $0.000007/query) and 100% (full T4 saturation = $0.0000007/query).

Why utilisation matters: a small team running a chatbot doesn't hit 200 inferences/second 24/7. The honest comparison number depends on your actual load. Reporting both ends bounds the answer. The calculator page also exposes a utilisation slider for the same reason.

Headline: Sonnet uncached at $0.00525 vs DistilBERT at realistic load $0.0000069 = **~756× cheaper**. At full utilisation, 7,560×.

> **Reconciled 2026-07-28.** This was the only place in the repo that stated the ratio correctly.
> Elsewhere it appeared as 750× (README, calculator) and as "500× at full utilisation, ~50× at low
> utilisation" (notebook 04 markdown) — four numbers for one quantity, with the utilisation label
> inverted in the last one. The repo now quotes a single figure: **~756×, at 10% GPU utilisation,
> against the uncached API, at list prices retrieved 2026-05-24.**
>
> Note also that two of the three inputs on the in-house side are assumptions with no artifact
> behind them: the 200 inferences/sec throughput and the 10% utilisation. The API-side token
> counts are estimates too — the rebuilt harness records `usage` per call, so that half becomes
> measured.

## Section 5 — break-even chart

Two cost models:
- **API-only:** $/month = queries/month × per-query rate. Linear in queries.
- **Self-hosted:** $/month = $360 fixed (always-on T4 = $0.50 × 24 × 30) + negligible per-query marginal. Effectively flat.

Plotted on log-log axes from 100 queries/month to 10 million.

Break-even points (where the lines cross):
- vs uncached Sonnet: 68,571 queries/month
- vs cached Sonnet: 244,898 queries/month

The vertical reference lines + small annotations make the break-even numbers readable without zooming.

Two things to call out about the chart:
- **Log-log is the right choice** because both axes span 5+ orders of magnitude. Linear would compress everything that matters.
- **The cached-Sonnet break-even is meaningfully higher** (245K vs 69K). If a team is already using prompt caching, they need more volume to justify self-hosting. Worth knowing — the calculator toggles this on/off.

## Section 6 — decision table

For five common business scenarios (hobby / small B2B / mid-market fintech / scaled product / public-cloud SaaS), the cell:
1. Picks the closest training-set size we measured (or interpolates — actually just nearest)
2. Looks up the ensemble macro-F1 at that n
3. Checks if it meets the Sonnet baseline (0.8913)
4. Computes API monthly + self-host monthly
5. Generates one of three recommendation strings:
   - **"API (quality not yet competitive at this n)"** when DistilBERT's macro-F1 < 0.8913
   - **"API (volume too low to justify self-host fixed cost)"** when quality is fine but $360 fixed > $X API marginal at this volume
   - **"Self-host (saves $Y/mo; labels pay back in Z mo)"** when both conditions for switching are met

The table writes to `docs/cost-decision-table.md` as a markdown table the README links to. Same data the calculator page uses to drive its verdict box — written from the same source so the two artefacts can't drift out of sync.

Why these five scenarios specifically: they cover the practitioner spectrum. "Hobby" anchors the low end (API obviously wins). "Public-cloud SaaS" anchors the high end (self-host saves $262K/month). "Mid-market fintech at n=2,500, 500K queries/month" is the inflection point that most readers will recognise as their actual situation.

## Section 7 — failure-mode overlap

The most analytically interesting section. For each of 3,080 test queries, who's right: Sonnet, DistilBERT (n=9,000 ensemble), both, or neither?

Computed via a merge on the `text` column — joins the n=9000_seed{0..4} ensemble predictions against Sonnet's predictions, both against the true labels. Result:

- Both correct: 2,646 (85.9%)
- DistilBERT-only correct: 231 (7.5%)
- Sonnet-only correct: 120 (3.9%)
- **Both wrong: 83 (2.7%)**

The interesting finding is the "both wrong" number. If the two models failed independently, the expected count would be `(1 - P_ft_correct) × (1 - P_llm_correct) × 3080 ≈ 21`. Observed 83 is **4× higher than independence predicts**. That's direct evidence of correlated failure — both models miss the same queries. Most likely explanation: Banking77 has its own label noise (~3% of queries have labels the models reasonably disagree with), and that floor is what both approaches hit.

The notebook surfaces this explicitly: "Observed >> expected: failures are CORRELATED → Banking77 noise is the shared floor." That's a sophisticated observation for the writeup — it tells readers that pushing past macro-F1 = 0.97 is essentially impossible on this dataset *regardless of model*, because the dataset's own labels disagree with themselves at the ~3% level.

The companion stacked-bar figure (`failure_overlap.png`) visualises the four buckets at a glance with percentages baked in.

The DistilBERT-only-correct count (231) being ~2× the Sonnet-only-correct (120) is a side observation — suggests somewhat anti-correlated failures between the two systems beyond the shared noise floor, which means a meta-ensemble of LLM + fine-tune as a tiebreaker could plausibly push past 92% accuracy. Mentioned in the README's failure-overlap section as "what's next if anyone wanted to," not pursued in this project.

## Section 8 — analysis_summary.json

Pulls every number computed in sections 2–7 into a single structured JSON: dataset metadata, LLM baseline, fine-tune quality (single-seed mean + ensemble + crossover n + peak), bootstrap CIs at every n, cost per inference for all four scenarios, break-even queries/month, failure-overlap counts.

Written to `results/analysis_summary.json` (~2.5 KB). Two consumers:

- **The README** quotes specific numbers from this file (the headline TL;DR, the decision table, the failure-overlap percentages). If those numbers ever change because of a re-run, this JSON is the place to look first.
- **The calculator page** (`docs/index.html`) hard-codes the same quality-by-n curve into its JavaScript. The values match what's in this JSON — kept in sync by hand, but the JSON is the canonical source.

Having a single structured output makes the project's claims auditable. A reader who wants to verify "DistilBERT at n=2,500 ensemble = 0.910" doesn't have to re-run anything — they open the JSON.

---

## How this notebook fits the project

Notebooks 01–03 generate raw data: training subsets, LLM predictions, 40 fine-tune runs. Notebook 04 is where that raw data becomes the project's argument. The crossover plot answers "does the fine-tune actually beat the LLM and at what n?". The bootstrap test answers "is that win real?". The cost analysis answers "should anyone care?". The failure-overlap answers "is there a deeper ceiling I should know about?".

The README + calculator + LinkedIn post all reference numbers and figures generated here. If you ever change the v2 recipe and re-run the sweep, the only thing that needs editing is `RECIPE` in notebook 03 — notebook 04 re-runs unchanged and re-generates every downstream artefact automatically.
