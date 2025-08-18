from pathlib import Path
from typing import Dict


def parse_intents_md(path: Path) -> Dict[str, str]:
    """Parse INTENTS.md and return mapping of intent_type to category.

    Categories starting with ``UNSUPPORTED`` are normalized to ``UNCLEAR_INTENT``
    so that both scripts and tests share the same logic.
    """
    intents: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("| ---") or "Intent Type" in line:
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 2:
            continue
        intent, category = parts[0], parts[1]
        if category.startswith("UNSUPPORTED"):
            category = "UNCLEAR_INTENT"
        intents[intent] = category
    return intents
