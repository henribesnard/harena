"""Entity extraction agent with optional teachability support.

The agent leverages :class:`AssistantAgent` from the autogen library. It can be
extended with normalizer callables that post-process extracted entities. When
the :class:`Teachability` capability is available, it is added to the agent so
that users can teach new entities during runtime.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional

try:  # pragma: no cover - resolved in tests via stub
    from autogen import AssistantAgent
except Exception:  # pragma: no cover - fallback
    class AssistantAgent:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def add_capability(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            pass

try:  # pragma: no cover - Teachability may not be installed
    from autogen.agentchat.contrib.capabilities.teachability import Teachability
except Exception:  # pragma: no cover - fallback stub when autogen extras absent
    class Teachability:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


class EntityExtractorAgent(AssistantAgent):
    """Agent chargé d'extraire des entités dans un texte."""

    def __init__(
        self,
        name: str = "entity_extractor",
        normalizers: Optional[Iterable[Callable[[str], str]]] = None,
        enable_teachability: bool = True,
        system_message: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        system_message = system_message or "Extract financial entities from user messages."
        super().__init__(name=name, system_message=system_message, **kwargs)

        self.normalizers: List[Callable[[str], str]] = list(normalizers or [])
        self.teachability: Optional[Teachability] = None

        if enable_teachability:
            try:
                self.teachability = Teachability()
                if hasattr(self, "add_capability"):
                    self.add_capability(self.teachability)
            except Exception:  # pragma: no cover - safety if capability not available
                self.teachability = None

    # ------------------------------------------------------------------
    # Normalisation des entités
    # ------------------------------------------------------------------
    def add_normalizer(self, normalizer: Callable[[str], str]) -> None:
        """Ajoute une fonction de normalisation d'entité."""
        self.normalizers.append(normalizer)

    def normalize(self, entity: str) -> str:
        """Applique tous les normalisateurs configurés à une entité."""
        for normalizer in self.normalizers:
            entity = normalizer(entity)
        return entity

    def normalize_entities(self, entities: Iterable[str]) -> List[str]:
        """Normalise une collection d'entités."""
        return [self.normalize(e) for e in entities]
