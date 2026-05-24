"""Score the Claude Code response against ground truth.

Reads results/sanity_test/response.json (Claude's JSON output, pasted back)
and compares to results/sanity_test/truth.json. Reports:
  - Parse status: did response.json deserialise?
  - Format check: right length, all required keys present
  - Hallucination check: all intent names in the 77-class list
  - Accuracy: % correct vs ground truth

If this scores 8+/10 with clean parse and zero hallucinations, Claude Code
via Sonnet 4.6 is viable for the full eval. If it scores poorly or hallucinates,
the prompt needs tightening before we scale to 300 dev rows.

Run: .venv/bin/python scripts/score_sanity_test.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_label_names

DIR = Path(__file__).resolve().parent.parent / "results" / "sanity_test"


def parse_response(text: str):
    """Try plain JSON first; if that fails, strip a markdown code fence."""
    text = text.strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        pass
    m = re.search(r"```(?:json)?\s*\n?(\[.*?\])\s*\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1)), "stripped markdown code fence"
        except json.JSONDecodeError as e:
            return None, f"JSON inside fence still invalid: {e}"
    return None, "no valid JSON found (no code fence either)"


def main() -> int:
    response_path = DIR / "response.json"
    truth_path = DIR / "truth.json"

    if not truth_path.exists():
        print(f"FAIL: {truth_path} not found. Run sanity_test_claude_code.py first.")
        return 1
    if not response_path.exists():
        print(f"FAIL: {response_path} not found.")
        print("Paste Claude's JSON response into that file, then re-run this.")
        return 1

    truth = json.loads(truth_path.read_text())
    response, note = parse_response(response_path.read_text())

    if response is None:
        print(f"FAIL: could not parse response — {note}")
        return 1

    if note:
        print(f"WARN: {note} (parsed anyway)")

    labels = set(load_label_names())
    truth_by_id = {t["id"]: t for t in truth}

    issues: list[str] = []
    if not isinstance(response, list):
        issues.append(f"Response is {type(response).__name__}, expected list")
    elif len(response) != len(truth):
        issues.append(f"Response has {len(response)} items, expected {len(truth)}")

    correct = 0
    hallucinated = 0

    print()
    print(f"  {'id':<3} {'truth':<40} {'predicted':<40} verdict")
    print(f"  {'-' * 3} {'-' * 40} {'-' * 40} {'-' * 12}")
    for item in (response if isinstance(response, list) else []):
        if not isinstance(item, dict):
            issues.append(f"Item is {type(item).__name__}, expected dict: {item!r}")
            continue
        if "id" not in item or "intent" not in item:
            issues.append(f"Item missing keys: {item}")
            continue

        item_id = item["id"]
        pred = str(item["intent"]).strip()
        t = truth_by_id.get(item_id)

        if t is None:
            print(f"  {item_id!s:<3} (no matching truth row)               {pred:<40} MISMATCH")
            continue

        truth_intent = t["intent"]
        if pred not in labels:
            hallucinated += 1
            verdict = "HALLUCINATED"
        elif pred == truth_intent:
            correct += 1
            verdict = "correct"
        else:
            verdict = "wrong"

        print(f"  {item_id!s:<3} {truth_intent:<40} {pred:<40} {verdict}")

    print()
    print(f"Accuracy:        {correct} / {len(truth)} ({100 * correct / max(len(truth), 1):.0f}%)")
    print(f"Hallucinations:  {hallucinated}")
    print(f"Format issues:   {len(issues)}")
    for x in issues:
        print(f"  - {x}")

    print()
    if correct >= 8 and hallucinated == 0 and not issues:
        print("PASS: Claude Code via Sonnet 4.6 looks viable for the full eval.")
        return 0
    if hallucinated > 0:
        print("FAIL: hallucinated intent names. Prompt needs stricter formatting.")
        return 2
    if issues:
        print("WARN: format issues. Prompt's JSON instructions need to be tighter.")
        return 2
    print("WARN: low accuracy. Could be a noisy 10-row sample — re-run or scale up to see.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
