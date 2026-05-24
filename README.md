# Small fine-tuned classifier vs LLM-as-judge — when does the cheap model win?

> *Production LLM apps are quietly replacing GPT-4 calls with fine-tuned small models because GPT-4 is expensive and slow. But almost nobody has rigorously characterised when the small model actually wins, how much training data it takes to get there, and what the cost/latency tradeoff looks like. This project does that systematically on a public classification task.*

**Status:** in progress · *Louis Grochla, May–June 2026*
**Audience:** AI/ML engineers deciding whether to fine-tune a small model or pay per-call to a frontier LLM; data science Masters admissions readers interested in evaluation methodology.

---

## TL;DR (placeholder — fill in once the experiment runs)

> *Headline result goes here once the fine-tune sweep is done. Example shape: "On the Banking77 intent-classification task (77 fine-grained intents from real banking customer queries), a fine-tuned DistilBERT (66M parameters) matches Claude Sonnet 4.6's macro-F1 at **N = X training examples**, while running at **~Y× lower cost per inference** and **~Z× lower latency**. Below that data threshold, the LLM still wins. Above it, the small model dominates on cost without sacrificing quality."*

---

## Why this question

When a team builds an LLM-powered product, the default move is to call a frontier API (Claude, GPT-4) on every request. This works until the bill arrives. The standard alternative — fine-tune a small open-source model on labelled examples — sounds cheaper, but raises real questions:

- **At what dataset size does the small model actually match the LLM?** No fine-tuned model with 50 examples is going to beat Claude. At 5,000? Maybe. At 50,000? Almost certainly. Where's the crossover?
- **What's the actual cost difference?** Self-hosting a model has fixed costs (GPU, deployment). The break-even isn't free.
- **How does latency compare?** A local fine-tuned model can serve at 10–50ms; a frontier API call is 1–3 seconds. When does that matter?
- **What about the long tail?** Frontier LLMs handle edge cases better. Does the small model fail catastrophically on rare inputs?

This study answers those questions on a real, reproducible benchmark.

## Why this dataset

**[Banking77](https://huggingface.co/datasets/PolyAI/banking77)** — 13,083 real customer queries from a banking chatbot, labelled into 77 fine-grained intents (e.g. `card_arrival`, `transfer_failed`, `lost_or_stolen_card`). Curated by PolyAI for the EACL 2020 paper *"Efficient Intent Detection with Dual Sentence Encoders."*

Three reasons this task is the right benchmark:

1. **It's a real commercial use case.** Every banking and fintech chatbot in production does intent classification at scale. LLMs are increasingly being deployed for it — and increasingly being replaced with fine-tuned small models when the volume justifies it. This question matters to the people building those systems.
2. **77 classes is non-trivial.** Most NLP benchmarks are binary or 3–5 class. 77 fine-grained classes — including genuinely confusable pairs like `card_payment_not_recognised` vs `card_payment_wrong_exchange_rate` — tests whether the small model can handle the long tail that frontier LLMs supposedly excel at.
3. **It's a structurally identical problem to salespatch's lead-qualification agent.** salespatch's pipeline includes an LLM stage that classifies UK independent businesses by industry and digital readiness from scraped text. Whatever crossover this study finds — at what *n* a small model matches the LLM, at what cost ratio — ports directly to that future fine-tune work.

## Approach

1. **Establish the LLM-as-judge baseline.** Run Claude Sonnet 4.6 on the Banking77 test set with a careful, versioned prompt. Iterate the prompt on a 300-row stratified dev slice (carved from the train pool, disjoint from test) to keep iteration cost bounded — roughly $1.50 per prompt version. The final prompt then runs once on the full 3,080-row test set (~$15) for the headline number.

2. **Fine-tune a small model at varying training-set sizes.** DistilBERT (66M params) at *n* ∈ {50, 100, 250, 500, 1000, 2500, 5000}. Fixed val set (500 rows carved from the train pool, same val for every n). **5 random seeds at each size for confidence intervals** — chosen over k-fold CV because at n=50 with 77 classes, k-fold puts most classes at zero examples per fold, which makes those metrics meaningless. Random seeds give the same variance estimate without the degenerate-fold problem.

3. **Plot the crossover.** Small-model performance vs *n*, with LLM baseline as a horizontal line. The intersection is the headline finding.

4. **Cost & latency analysis.** $/1k inferences for both approaches, ms/inference, GPU memory, payback period if you process X queries/month.

5. **Statistical significance test.** Paired bootstrap on the macro-F1 difference at the candidate crossover *n*. Report 95% CI on the gap — "the means look close" isn't the same as "the models actually match." A CI that crosses zero means we can't claim crossover at that *n*.

6. **Failure-mode analysis.** Where does each approach fail? Are the failures correlated (both miss the same hard cases) or anti-correlated (different weaknesses)? This matters for ensembling decisions in production.

## Deliverables

- `notebooks/01_data_prep.ipynb` — load Banking77, audit class balance + query length, carve fixed val + dev splits, sample stratified training subsets at each (n, seed)
- `notebooks/02_llm_baseline.ipynb` — versioned prompt loader, dev-slice iteration loop, final eval on the full 3,080-row test set, calibration plots
- `notebooks/03_finetune_sweep.ipynb` — DistilBERT fine-tuning at 7 dataset sizes × 5 random seeds
- `notebooks/04_analysis.ipynb` — headline figures: accuracy-vs-data crossover, cost analysis, paired-bootstrap significance test, failure-mode breakdown
- `app/gradio_demo.py` — interactive Gradio demo: paste a banking customer query, see Claude vs DistilBERT predictions side by side
- `src/` — reusable helpers (`eval.py` metrics + bootstrap, `prompts.py` loader, `data.py` split loaders) imported from the notebooks
- `src/prompts/` — versioned prompt text files, diff-able in git
- `results/figures/` — all final plots, exported as PNG
- `costs.md` — written-up cost analysis for AI builders weighing this tradeoff
- *(Stretch)* model weights published to [HuggingFace Hub](https://huggingface.co/louisgrochla) under MIT licence

## How to reproduce

```bash
git clone https://github.com/louisgrochla/ds-llm-judge-vs-small-model.git
cd ds-llm-judge-vs-small-model
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Banking77 loads automatically from HuggingFace on first run — no manual download needed
cp .env.example .env  # then paste your ANTHROPIC_API_KEY

jupyter lab
# Run notebooks in order: 01 → 02 → 03 → 04
```

GPU note: notebooks `03` and `04` run faster on a GPU. Free Google Colab (T4) is sufficient — total compute budget is roughly 70 minutes across the full fine-tune sweep.

## Limitations (to be expanded after the experiment)

- *(placeholder — list once the experiment surfaces them)*
- Banking77 is English-only — results may not generalise to multilingual customer service contexts
- Single-judge LLM baseline; production systems often ensemble multiple judges
- DistilBERT is one architecture choice; other small models (MiniLM, MobileBERT, MPNet) may behave differently
- Banking domain is relatively narrow — performance may vary on broader customer-service intent taxonomies (e.g. CLINC150's 150-class set)

## What I'd do next

- *(placeholder — fill in once results are in)*
- Repeat on a second dataset (e.g. CLINC150 for broader intent set with out-of-scope detection) to test methodology generalisation
- Add Claude Haiku 4.5 as a "mid-tier API" baseline between Sonnet and the fine-tune
- Investigate distilling Claude into the small model directly via teacher-forcing on Claude's outputs
- Port the methodology to my own production system ([salespatch](https://github.com/louisgrochla/salespatch)) once the dataset reaches *n* > 200

## Why I built this

I'm a final-year UK Business Analytics undergrad. Plan: graduate June 2027, gap-year UK DS/ML role, then US MS Data Science starting autumn 2028 (applications autumn 2027). My main project, [salespatch](https://github.com/louisgrochla/salespatch), is a multi-agent AI platform where LLMs classify and qualify UK independent businesses at multiple pipeline stages — and the planned next architectural step is to replace those LLM stages with smaller fine-tuned models once we have enough outcome data. My own dataset is still small. So I pre-ran the experiment on a structurally identical public task to characterise exactly when the small model wins. The methodology here ports directly to that future work.

## Contact

Louis Grochla — louisgrochla27@gmail.com · [LinkedIn](https://www.linkedin.com/in/louisgrochla/) · [GitHub](https://github.com/louisgrochla)
