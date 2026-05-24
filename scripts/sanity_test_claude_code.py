"""Generate the sanity-test prompt for Claude Code Sonnet 4.6 evaluation.

Pulls 10 stratified queries from the dev slice, builds the full 77-intent
prompt, and writes everything you need to results/sanity_test/:
  - prompt.txt   the thing you paste into Claude Code
  - truth.json   ground-truth labels for scoring after paste-back

Run: .venv/bin/python scripts/sanity_test_claude_code.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_dev_slice, load_label_names

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "sanity_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_QUERIES = 10
SEED = 42

dev = load_dev_slice()
labels = load_label_names()

# Sample 10 rows. Bias toward variety — pull from different label buckets
# so we exercise common, rare, and confusable intents in one go.
sample = dev.sample(n=N_QUERIES, random_state=SEED).reset_index(drop=True)

label_list_block = "\n".join(f"- {name}" for name in labels)
query_block = "\n".join(f"{i + 1}. {row['text']}" for i, row in sample.iterrows())

prompt = f"""You are an intent classifier for a banking customer-service chatbot.

For each user query below, assign it to exactly one of the following 77 intents:

{label_list_block}

User queries (numbered):

{query_block}

Return ONLY a JSON array, one object per query, in the same order. Each object must have exactly two keys:
- "id": the query number (1 to {N_QUERIES})
- "intent": the exact intent name from the list above (lowercase with underscores, no surrounding quotes inside the value)

No commentary. No markdown code fences. No explanation. Just the JSON array starting with [ and ending with ].
"""

prompt_path = OUT_DIR / "prompt.txt"
prompt_path.write_text(prompt)

truth = [
    {"id": i + 1, "intent": labels[int(row["label"])], "text": row["text"]}
    for i, row in sample.iterrows()
]
truth_path = OUT_DIR / "truth.json"
truth_path.write_text(json.dumps(truth, indent=2))

print(f"Wrote {prompt_path}")
print(f"Wrote {truth_path}")
print()
print(f"Prompt is {len(prompt):,} chars, ~{len(prompt) // 4:,} tokens (Sonnet handles this easily).")
print()
print("Next steps:")
print(f"  1. cat {prompt_path}   # or open it in your editor and select-all + copy")
print(f"  2. Open a FRESH terminal in any directory (NOT this repo, to keep context clean)")
print(f"  3. Run: claude")
print(f"  4. Once Claude Code starts, run: /model claude-sonnet-4-6")
print(f"     (verify status bar shows Sonnet 4.6)")
print(f"  5. Paste the prompt, hit enter")
print(f"  6. Copy Claude's response (just the JSON array)")
print(f"  7. Save it to: {OUT_DIR}/response.json")
print(f"  8. Run: .venv/bin/python scripts/score_sanity_test.py")
