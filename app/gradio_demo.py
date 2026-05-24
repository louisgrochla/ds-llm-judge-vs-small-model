"""Gradio demo: in-house DistilBERT vs Claude Sonnet 4.6 on Banking77.

Side-by-side prediction with live DistilBERT latency. Sonnet predictions cached
from the project's test-set eval (notebook 02). No API key required.

Run locally:    python app/gradio_demo.py
HF Spaces:      rename to app.py, set hardware to CPU.
"""

import json
import time
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# === Model + data loading =====================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_MODEL_PATH = REPO_ROOT / "results" / "checkpoints" / "v2_retune_n9000_seed0"
MODEL_SOURCE = str(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH.exists() else "louisgrochla/distilbert-banking77"

print(f"Loading model from: {MODEL_SOURCE}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_SOURCE)
model.eval()

LABEL_NAMES_PATH = REPO_ROOT / "data" / "processed" / "label_names.txt"
LABEL_NAMES = (
    LABEL_NAMES_PATH.read_text().splitlines()
    if LABEL_NAMES_PATH.exists()
    else [model.config.id2label[i] for i in range(model.config.num_labels)]
)

# Warm up the model so the first user-facing inference isn't a cold start (~470 ms → ~25 ms)
with torch.no_grad():
    _warm = tokenizer("warmup", truncation=True, padding="max_length", max_length=64, return_tensors="pt")
    _ = model(**_warm)

EXAMPLES = json.loads((Path(__file__).resolve().parent / "demo_examples.json").read_text())
EXAMPLE_LABELS = [f"[{e['category']}] {e['query']}" for e in EXAMPLES]
EXAMPLE_LOOKUP = {label: e for label, e in zip(EXAMPLE_LABELS, EXAMPLES)}

# === Cost constants (from notebook 04) ========================================

SONNET_COST_PER_QUERY = 0.0053          # USD, uncached
DISTILBERT_COST_PER_QUERY = 0.000007    # USD, T4 at realistic utilisation
SONNET_TYPICAL_LATENCY_MS = 1500


# === Inference ================================================================

def classify_distilbert(query: str) -> tuple[str, float]:
    start = time.perf_counter()
    enc = tokenizer(query, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    pred_id = int(logits.argmax(dim=-1).item())
    return LABEL_NAMES[pred_id], (time.perf_counter() - start) * 1000


# === Rendering ================================================================

CSS = """
.gradio-container { max-width: 1080px !important; }

#query-box {
  background: #f8f9fb;
  border-left: 4px solid #94a3b8;
  padding: 14px 20px;
  margin: 18px 0 12px 0;
  font-size: 17px;
  font-style: italic;
  border-radius: 4px;
  color: #1f2937;
}

.card {
  padding: 22px 24px;
  border-radius: 12px;
  background: #ffffff;
  border: 1px solid #e5e7eb;
  min-height: 200px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.card-sonnet { border-top: 4px solid #dc2626; }
.card-distil { border-top: 4px solid #2563eb; }

.card-label {
  font-size: 11px;
  color: #6b7280;
  text-transform: uppercase;
  letter-spacing: 0.7px;
  font-weight: 600;
  margin-bottom: 14px;
}

.prediction {
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 19px;
  font-weight: 600;
  margin: 0 0 18px 0;
  color: #111827;
  word-break: break-all;
}

.correct { color: #059669; font-size: 18px; margin-left: 6px; }
.wrong   { color: #dc2626; font-size: 18px; margin-left: 6px; }

.metric {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 9px 0;
  border-top: 1px solid #f3f4f6;
}
.metric:first-of-type { border-top: none; padding-top: 0; }

.metric-label { color: #6b7280; font-size: 13px; }
.metric-value { font-weight: 600; font-size: 14px; color: #111827; }

#ratio-banner {
  background: #f0f7ff;
  border: 1px solid #c7dcfa;
  border-radius: 12px;
  padding: 22px;
  margin: 18px 0 6px 0;
  text-align: center;
  font-size: 16px;
  color: #1f2937;
}

#ratio-banner .big {
  font-size: 30px;
  font-weight: 700;
  color: #2563eb;
  margin: 0 4px;
  font-variant-numeric: tabular-nums;
}

#footnote {
  text-align: center;
  font-size: 13px;
  color: #6b7280;
  margin-top: 4px;
}
#footnote code { background: #f3f4f6; padding: 2px 6px; border-radius: 3px; font-size: 12px; }
"""


def render_card(name: str, subtitle: str, color_class: str, intent: str, is_correct: bool,
                cost_usd: float, latency_str: str, training_data: str) -> str:
    mark = '<span class="correct">✓</span>' if is_correct else '<span class="wrong">✗</span>'
    cost_str = f"${cost_usd:.4f}" if cost_usd >= 0.0001 else f"${cost_usd:.6f}"
    return (
        f'<div class="card {color_class}">'
        f'<div class="card-label">{name} · {subtitle}</div>'
        f'<div class="prediction">{intent}{mark}</div>'
        f'<div class="metric"><span class="metric-label">cost per query</span><span class="metric-value">{cost_str}</span></div>'
        f'<div class="metric"><span class="metric-label">latency</span><span class="metric-value">{latency_str}</span></div>'
        f'<div class="metric"><span class="metric-label">training data</span><span class="metric-value">{training_data}</span></div>'
        f'</div>'
    )


def render_banner(cost_ratio: float, latency_ratio: float) -> str:
    return (
        f'<div id="ratio-banner">'
        f'DistilBERT was <span class="big">{cost_ratio:,.0f}×</span> cheaper and '
        f'<span class="big">{latency_ratio:,.0f}×</span> faster on this query'
        f'</div>'
    )


def predict_example(example_label: str):
    if not example_label:
        return "", "", "", "", ""

    example = EXAMPLE_LOOKUP[example_label]
    query = example["query"]
    truth = example["true_intent"]
    sonnet_pred = example["sonnet_pred"]

    distil_pred, distil_latency = classify_distilbert(query)

    query_html = f'<div id="query-box">"{query}"</div>'

    sonnet_card = render_card(
        "Claude Sonnet 4.6", "zero-shot via API", "card-sonnet",
        sonnet_pred, sonnet_pred == truth,
        SONNET_COST_PER_QUERY, f"~{SONNET_TYPICAL_LATENCY_MS:,} ms", "0 examples",
    )

    distil_card = render_card(
        "DistilBERT (in-house)", "fine-tuned, live on CPU", "card-distil",
        distil_pred, distil_pred == truth,
        DISTILBERT_COST_PER_QUERY, f"{distil_latency:,.0f} ms", "9,000 examples",
    )

    banner = render_banner(
        SONNET_COST_PER_QUERY / DISTILBERT_COST_PER_QUERY,
        SONNET_TYPICAL_LATENCY_MS / max(distil_latency, 1),
    )

    footnote = (
        f'<div id="footnote">'
        f'Banking77 ground-truth label: <code>{truth}</code> · {example["category"]}'
        f'</div>'
    )

    return query_html, sonnet_card, distil_card, banner, footnote


# === UI =======================================================================

with gr.Blocks(title="DistilBERT vs Claude Sonnet 4.6 — Banking77") as demo:
    gr.Markdown("## In-house DistilBERT vs Claude Sonnet 4.6 — Banking77")
    gr.Markdown(
        "Banking customer-service intent classification, 77 classes. "
        "Fine-tuned DistilBERT matches Sonnet at n=2,500 training examples, beats it at n=5,000+. "
        "[Methodology & source](https://github.com/louisgrochla/ds-llm-judge-vs-small-model)"
    )

    with gr.Row():
        example_dd = gr.Dropdown(
            choices=EXAMPLE_LABELS,
            label="Try a banking customer query",
            value=EXAMPLE_LABELS[0],
            scale=4,
        )
        classify_btn = gr.Button("Compare", variant="primary", scale=1)

    query_html_out = gr.HTML()
    with gr.Row():
        sonnet_html_out = gr.HTML()
        distil_html_out = gr.HTML()
    banner_html_out = gr.HTML()
    footnote_html_out = gr.HTML()

    with gr.Accordion("How this demo works", open=False):
        gr.Markdown(
            """
            **Sonnet predictions** are cached from the project's test-set evaluation (3,080 queries via Claude Code on a Max subscription — no API spend). For the 12 example queries here, you're seeing the exact prediction the LLM gave on the test set.

            **DistilBERT runs live** on whatever CPU the demo is hosted on. The latency you see is real measured milliseconds. Cost is calculated for HF Inference Endpoint T4 at realistic utilisation — see notebook 04 in the repo for the full cost analysis.

            **Why no free-form input box:** for arbitrary queries DistilBERT would still work, but Sonnet would need a live API call (and someone to pay for it). The 12 examples are pre-curated to cover all 4 outcome categories — both right, DistilBERT only, Sonnet only, both wrong — making the build-vs-buy case viscerally without typing.

            **Banking77** has measurable label noise (~3% of queries have debatable labels by the dataset creators' own analysis). The "both wrong" examples here are mostly cases where both models pick the obviously-correct intent and Banking77's label is the one in disagreement.
            """
        )

    classify_btn.click(
        predict_example, inputs=[example_dd],
        outputs=[query_html_out, sonnet_html_out, distil_html_out, banner_html_out, footnote_html_out],
    )
    demo.load(
        predict_example, inputs=[example_dd],
        outputs=[query_html_out, sonnet_html_out, distil_html_out, banner_html_out, footnote_html_out],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CSS)
