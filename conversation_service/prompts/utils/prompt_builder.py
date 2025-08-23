"""Utilities for assembling prompts based on intent and context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, List


@dataclass(frozen=True)
class PromptSection:
    """A section of text that can participate in a prompt.

    Args:
        text: Raw text of the section. It may contain ``str.format`` placeholders.
        intents: Optional list of intents this section is relevant for. If ``None``
            the section is always included.
        priority: Sections with higher priority are placed earlier and kept when the
            prompt needs to be truncated to fit a length constraint.
    """

    text: str
    intents: Optional[Sequence[str]] = None
    priority: int = 0


class PromptBuilder:
    """Assemble prompts from multiple sections with caching and truncation."""

    def __init__(self, sections: Sequence[PromptSection], *, max_length: Optional[int] = None) -> None:
        self._sections = list(sections)
        self._max_length = max_length
        self._cache: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], str] = {}

    # ------------------------------------------------------------------
    def build(self, intent: str, context: Optional[Mapping[str, object]] = None) -> str:
        """Return the prompt for ``intent`` using ``context`` variables.

        Results are cached based on the intent and the provided context so that
        repeated calls with identical parameters do not recompute the prompt.
        """

        context_dict = dict(context or {})
        cache_key = (
            intent,
            tuple(sorted((str(k), str(v)) for k, v in context_dict.items())),
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # Select sections relevant for the intent and order by priority
        relevant = [s for s in self._sections if not s.intents or intent in s.intents]
        relevant.sort(key=lambda s: s.priority, reverse=True)

        parts: List[str] = []
        for section in relevant:
            try:
                rendered = section.text.format(**context_dict)
            except KeyError:
                # Missing variable; leave placeholder as-is
                rendered = section.text
            parts.append(rendered)

        prompt = "\n".join(parts)
        if self._max_length is not None and len(prompt) > self._max_length:
            prompt = prompt[: self._max_length]

        self._cache[cache_key] = prompt
        return prompt
