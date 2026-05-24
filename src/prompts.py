"""Prompt version loader. Each prompt version lives as a text file in src/prompts/."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(version: str) -> str:
    """Load a versioned prompt from src/prompts/{version}.txt.

    Returns the raw template string. Placeholders use Python str.format syntax
    (e.g. {query}, {label_list}). Literal braces in the template must be
    escaped as {{ and }}.
    """
    path = PROMPTS_DIR / f"{version}.txt"
    if not path.exists():
        available = sorted(p.stem for p in PROMPTS_DIR.glob("*.txt"))
        raise FileNotFoundError(
            f"Prompt version {version!r} not found in {PROMPTS_DIR}. Available: {available}"
        )
    return path.read_text()


def list_versions() -> list[str]:
    return sorted(p.stem for p in PROMPTS_DIR.glob("*.txt"))
