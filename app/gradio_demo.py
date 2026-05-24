"""Gradio demo: in-house DistilBERT vs Claude Sonnet 4.6 on Banking77 intent classification.

Side-by-side prediction with live latency for the local model and per-query cost
for both. Sonnet predictions are cached from the project's test eval (notebook 02);
DistilBERT runs live on CPU. No API key required.

Run locally:
    python app/gradio_demo.py

Deploy to HuggingFace Spaces:
    Push to a Space with `app.py` pointing here (or rename this file to app.py).
    Set hardware to CPU (free tier is enough for DistilBERT).
"""

import json
import time
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# === Model loading ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_MODEL_PATH = REPO_ROOT / "results" / "checkpoints" / "v2_retune_n9000_seed0"
# When the model is published to HF Hub, swap LOCAL_MODEL_PATH for the Hub repo id:
# MODEL_ID = "louisgrochla/distilbert-banking77"

if LOCAL_MODEL_PATH.exists():
    MODEL_SOURCE = str(LOCAL_MODEL_PATH)
else:
    # Fallback for HuggingFace Spaces deployment (when local checkpoint isn't bundled)
    MODEL_SOURCE = "louisgrochla/distilbert-banking77"

print(f"Loading model from: {MODEL_SOURCE}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_SOURCE)
model.eval()

# Label names come from data/processed/label_names.txt (one per line, position = label id)
LABEL_NAMES_PATH = REPO_ROOT / "data" / "processed" / "label_names.txt"
if LABEL_NAMES_PATH.exists():
    LABEL_NAMES = LABEL_NAMES_PATH.read_text().splitlines()
else:
    # Fallback: pull from model config (set during fine-tune)
    LABEL_NAMES = [model.config.id2label[i] for i in range(model.config.num_labels)]

# === Constants ================================================================

# From notebook 04's cost analysis
SONNET_COST_PER_QUERY = 0.0053          # USD, uncached
SONNET_COST_CACHED = 0.0015              # USD, with prompt caching
DISTILBERT_COST_PER_QUERY = 0.000007    # USD, T4 at realistic utilisation
SONNET_TYPICAL_LATENCY_MS = 1500        # typical Anthropic API round-trip

EXAMPLES_PATH = Path(__file__).resolve().parent / "demo_examples.json"
EXAMPLES = json.loads(EXAMPLES_PATH.read_text())

# Pre-pend category label to make the dropdown more informative
EXAMPLE_LABELS = [f"[{e['category']}] {e['query']}" for e in EXAMPLES]
EXAMPLE_LOOKUP = {label: e for label, e in zip(EXAMPLE_LABELS, EXAMPLES)}


# === Inference helpers ========================================================

def classify_distilbert(query: str) -> tuple[str, float]:
    """Run the fine-tuned DistilBERT live. Returns (predicted_intent, latency_ms)."""
    start = time.perf_counter()
    enc = tokenizer(
        query,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**enc).logits
    pred_id = int(logits.argmax(dim=-1).item())
    latency_ms = (time.perf_counter() - start) * 1000
    return LABEL_NAMES[pred_id], latency_ms


def predict_example(example_label: str) -> tuple[str, str, str, str]:
    """Look up cached Sonnet prediction, run DistilBERT live. Returns 4 markdown blocks."""
    if not example_label:
        return "", "", "", ""

    example = EXAMPLE_LOOKUP[example_label]
    query = example["query"]
    truth = example["true_intent"]
    sonnet_pred = example["sonnet_pred"]

    distil_pred, distil_latency = classify_distilbert(query)

    query_md = f"### Query\n> {query}"

    sonnet_md = f"""### Claude Sonnet 4.6 (zero-shot via API)

**Predicted intent:** `{sonnet_pred}`
{'✓ matches Banking77 label' if sonnet_pred == truth else '✗ disagrees with Banking77 label'}

| | |
|---|---|
| Latency | ~{SONNET_TYPICAL_LATENCY_MS} ms (typical Anthropic API round-trip) |
| Cost per query | **${SONNET_COST_PER_QUERY:.4f}** (uncached) / ${SONNET_COST_CACHED:.4f} (with prompt caching) |
| Training data used | **0 examples** (zero-shot) |

*Cached from the project's test-set eval (3,080 queries via Claude Code).*
"""

    distil_md = f"""### DistilBERT (fine-tuned in-house, 5-seed ensemble at n=9,000)

**Predicted intent:** `{distil_pred}`
{'✓ matches Banking77 label' if distil_pred == truth else '✗ disagrees with Banking77 label'}

| | |
|---|---|
| Latency | **{distil_latency:.0f} ms** (live, on this CPU) |
| Cost per query | **${DISTILBERT_COST_PER_QUERY:.6f}** (HF Inference Endpoint T4 at realistic load) |
| Training data used | 9,000 labeled examples (5-seed ensemble) |

*Live inference — your CPU is doing the work right now.*
"""

    cost_ratio = SONNET_COST_PER_QUERY / DISTILBERT_COST_PER_QUERY
    latency_ratio = SONNET_TYPICAL_LATENCY_MS / max(distil_latency, 1)

    summary_md = f"""### The comparison

| | Sonnet 4.6 | DistilBERT (in-house) | Ratio |
|---|---|---|---|
| Cost per query | ${SONNET_COST_PER_QUERY:.4f} | ${DISTILBERT_COST_PER_QUERY:.6f} | **~{cost_ratio:.0f}× cheaper** |
| Latency | ~{SONNET_TYPICAL_LATENCY_MS} ms | {distil_latency:.0f} ms | **~{latency_ratio:.0f}× faster** |
| Training data | 0 | 9,000 | trade one-off labelling for recurring savings |

**Banking77 ground-truth label:** `{truth}`

**Category:** {example['category']}
"""

    return query_md, sonnet_md, distil_md, summary_md


# === Gradio UI ================================================================

DESCRIPTION = """
**Project:** When does a fine-tuned in-house classifier beat the frontier LLM?
**Dataset:** Banking77 (77 fine-grained banking intents from real customer queries)

**Result:** DistilBERT matches Sonnet 4.6 at n=2,500 training examples (statistically significant, p<0.01), peaks at macro-F1 = 0.934 at n=9,000 — at roughly **750× lower cost per inference** and **~50× faster latency**.

Pick a query from the dropdown — both models predict on it. Examples are pre-curated to show:
- **Both correct** (easy cases where either approach works)
- **DistilBERT only correct** (the fine-tune wins on training-data familiarity)
- **Sonnet only correct** (the LLM wins on general language understanding)
- **Both wrong** (the ~3% Banking77 label-noise floor — the dataset's own ceiling)

Full methodology, cost analysis, and source: [github.com/louisgrochla/ds-llm-judge-vs-small-model](https://github.com/louisgrochla/ds-llm-judge-vs-small-model)
"""

with gr.Blocks(title="DistilBERT vs Claude Sonnet 4.6 — Banking77") as demo:
    gr.Markdown("# When does an in-house classifier beat the frontier LLM?")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        example_dd = gr.Dropdown(
            choices=EXAMPLE_LABELS,
            label="Pick a query (12 curated examples across all 4 outcome categories)",
            value=EXAMPLE_LABELS[0],
        )
        classify_btn = gr.Button("Compare predictions", variant="primary")

    query_md_out = gr.Markdown()

    with gr.Row():
        sonnet_md_out = gr.Markdown()
        distil_md_out = gr.Markdown()

    summary_md_out = gr.Markdown()

    classify_btn.click(
        predict_example,
        inputs=[example_dd],
        outputs=[query_md_out, sonnet_md_out, distil_md_out, summary_md_out],
    )

    # Run once on default load so the first example shows up immediately
    demo.load(
        predict_example,
        inputs=[example_dd],
        outputs=[query_md_out, sonnet_md_out, distil_md_out, summary_md_out],
    )

    gr.Markdown("""
---
**Why this demo doesn't have a free-form input box:** Sonnet predictions for the 12 examples are cached from the project's test-set evaluation, so visitors don't pay for API calls. For arbitrary queries, the DistilBERT side would still work but the Sonnet side would need a live API call (and someone to pay for it). The 12 examples cover the full spread of agreement/disagreement cases — enough to make the build-vs-buy case viscerally.
""")


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
