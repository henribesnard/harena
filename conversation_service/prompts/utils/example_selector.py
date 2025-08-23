from __future__ import annotations

"""Utilities to select relevant and diverse few-shot examples."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


INTENTS_FILE = Path(__file__).resolve().parents[3] / "INTENTS.md"


def _load_intent_groups(path: Path) -> Dict[str, str]:
    """Map each intent to its high level group from ``INTENTS.md``."""
    mapping: Dict[str, str] = {}
    current_group: Optional[str] = None
    if not path.exists():
        return mapping

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("##"):
                # ``## 1. Transactions`` -> ``Transactions``
                try:
                    current_group = line.split(".", 1)[1].strip()
                except IndexError:  # pragma: no cover - defensive
                    current_group = line.lstrip("# ")
            elif current_group and line.startswith("|") and not line.startswith("| Intent"):
                parts = [p.strip() for p in line.strip("|").split("|")]
                if parts and parts[0]:
                    mapping[parts[0]] = current_group
    return mapping


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@dataclass
class Example:
    """A single training example."""

    text: str
    intent: str
    group: str
    embedding: np.ndarray
    score: float = 0.0
    usage_count: int = 0


class ExampleSelector:
    """Select few-shot examples based on semantic similarity.

    The selector ensures diversity by balancing examples across intent groups
    defined in ``INTENTS.md`` and by rotating frequently used examples.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        state_path: Optional[Path] = None,
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.intent_groups = _load_intent_groups(INTENTS_FILE)
        self.examples: List[Example] = []
        self.state_path = state_path
        if state_path and state_path.exists():
            self.load_state()

    # ------------------------------------------------------------------
    # Example management
    # ------------------------------------------------------------------
    def add_examples(self, items: Iterable[Tuple[str, str]]) -> None:
        """Add a collection of ``(text, intent)`` pairs."""
        for text, intent in items:
            group = self.intent_groups.get(intent, "UNKNOWN")
            emb = self.model.encode(text)
            self.examples.append(Example(text=text, intent=intent, group=group, embedding=emb))

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def select(self, query: str, k: int = 5) -> List[Example]:
        """Select ``k`` relevant examples for ``query``.

        Selection is performed by cosine similarity with embeddings and a
        round-robin pick across intent groups to guarantee diversity.
        """
        if not self.examples:
            return []

        query_emb = self.model.encode(query)
        for ex in self.examples:
            ex.score = _cosine(query_emb, ex.embedding) - 0.01 * ex.usage_count

        # Group examples by intent group and sort within each group by score
        grouped: Dict[str, List[Example]] = {}
        for ex in self.examples:
            grouped.setdefault(ex.group, []).append(ex)
        for items in grouped.values():
            items.sort(key=lambda e: e.score, reverse=True)

        # Round-robin selection to keep groups balanced
        selected: List[Example] = []
        groups = list(grouped.items())
        idx = 0
        while len(selected) < k and groups:
            _, items = groups[idx]
            if items:
                ex = items.pop(0)
                ex.usage_count += 1
                selected.append(ex)
            if not items:
                groups.pop(idx)
                if not groups:
                    break
                idx %= len(groups)
            else:
                idx = (idx + 1) % len(groups)

        if self.state_path:
            self.save_state()
        return selected

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_state(self) -> None:
        """Persist examples to JSON for rotation across sessions."""
        if not self.state_path:
            return
        data = [
            {
                "text": e.text,
                "intent": e.intent,
                "group": e.group,
                "embedding": e.embedding.tolist(),
                "usage_count": e.usage_count,
            }
            for e in self.examples
        ]
        with self.state_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

    def load_state(self) -> None:
        """Load examples from a JSON state file."""
        if not self.state_path or not self.state_path.exists():
            return
        with self.state_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.examples = [
            Example(
                text=item["text"],
                intent=item["intent"],
                group=item.get("group", "UNKNOWN"),
                embedding=np.array(item["embedding"], dtype=float),
                usage_count=item.get("usage_count", 0),
            )
            for item in data
        ]
