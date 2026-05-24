"""Batch-prompt construction and response parsing for the LLM-as-judge in notebook 02."""

import json
import re
from typing import Optional


def build_batch_prompt(
    template: str,
    queries: list[str],
    label_names: list[str],
    start_id: int = 1,
) -> str:
    """Fill a prompt template (loaded from src/prompts/vN.txt) with one batch of queries.

    The template must contain `{label_list_block}`, `{query_block}`, `{start_id}`, `{end_id}`.
    """
    label_list_block = "\n".join(f"- {name}" for name in label_names)
    query_block = "\n".join(f"{start_id + i}. {q}" for i, q in enumerate(queries))
    end_id = start_id + len(queries) - 1
    return template.format(
        label_list_block=label_list_block,
        query_block=query_block,
        start_id=start_id,
        end_id=end_id,
    )


def parse_response(text: str) -> tuple[Optional[list[dict]], Optional[str]]:
    """Parse Claude's JSON-array response, tolerant of common wrappers.

    Tries in order:
      1. Plain JSON
      2. JSON inside a ```json ... ``` markdown fence
      3. The first `[ ... ]` substring that parses as JSON

    Returns `(parsed_list, warning_or_none)`. On failure: `(None, error_message)`.
    """
    text = text.strip()

    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*\n?(\[.*?\])\s*\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1)), "stripped markdown code fence"
        except json.JSONDecodeError as e:
            return None, f"JSON inside markdown fence still invalid: {e}"

    m = re.search(r"(\[\s*\{.*?\}\s*\])", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1)), "extracted JSON array from surrounding text"
        except json.JSONDecodeError as e:
            return None, f"could not parse extracted array: {e}"

    return None, "no valid JSON array found in response"
