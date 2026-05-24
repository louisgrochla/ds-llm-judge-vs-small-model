"""Gradio demo — paste a banking customer query, see Claude vs fine-tuned DistilBERT predictions side by side.

Run: python app/gradio_demo.py
Deploy: push to Hugging Face Spaces (free), or `share=True` for a temporary public URL.
"""

import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: from src.prompts import load_prompt    # PROMPT = load_prompt('v_final')
# TODO: from src.data import load_label_names  # LABELS = load_label_names()
# TODO: load the fine-tuned DistilBERT checkpoint from the best (n, seed) run in results/


def predict(query: str):
    """Return both predictions side by side."""
    # TODO: implement predict_llm(query) -> (intent, latency_ms, cost_usd)
    # TODO: implement predict_small(query) -> (intent, latency_ms)
    llm_intent = "TODO"
    llm_latency_ms = 0
    llm_cost_usd = 0.0
    small_intent = "TODO"
    small_latency_ms = 0

    return {
        "Claude Sonnet 4.6": f"Intent: {llm_intent}\nLatency: {llm_latency_ms} ms\nCost: ${llm_cost_usd:.5f}",
        "DistilBERT (fine-tuned on Banking77)": f"Intent: {small_intent}\nLatency: {small_latency_ms} ms\nCost: ~$0 (self-hosted)",
    }


EXAMPLES = [
    "I noticed an extra fee when I withdrew money.",
    "How long does a transfer to a different bank usually take?",
    "My card was declined when I tried to pay.",
    "I think someone has stolen my card details.",
    "Can I top up using Apple Pay?",
]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Banking customer query", lines=3, placeholder="e.g. My card was declined when I tried to pay..."),
    outputs=gr.JSON(label="Predictions"),
    examples=EXAMPLES,
    title="LLM vs fine-tuned small model — banking intent classification",
    description=(
        "Compare Claude Sonnet 4.6 against a DistilBERT model fine-tuned on Banking77. "
        "See https://github.com/louisgrochla/ds-llm-judge-vs-small-model for the methodology and findings."
    ),
)


if __name__ == "__main__":
    demo.launch()
