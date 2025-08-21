"""Simple agent pipeline execution utilities.

This module exposes the :class:`AgentPipeline` which executes a sequence
of callable agents.  Each step receives the output of the previous step
and can be synchronous or asynchronous.  An optional
:class:`~core.fallback_manager.FallbackManager` can be used to provide
fallback behaviours for individual steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from .fallback_manager import FallbackManager

CallableType = Callable[..., Any]


@dataclass
class PipelineStep:
    """Represent a single step in a pipeline.

    Attributes
    ----------
    main:
        The primary callable for this step.
    fallbacks:
        Optional sequence of fallback callables executed if ``main``
        raises an exception.
    """

    main: CallableType
    fallbacks: Sequence[CallableType] | None = None


class AgentPipeline:
    """Run a series of agents sequentially."""

    def __init__(self, steps: Iterable[PipelineStep]) -> None:
        self.steps = list(steps)
        self.fallback_manager = FallbackManager()

    async def run(self, data: Any) -> Any:
        """Execute the pipeline starting with ``data``.

        The output of each step becomes the input for the next one.  If a
        step defines fallbacks, the :class:`FallbackManager` will try them
        in order.
        """
        result = data
        for step in self.steps:
            handlers = [step.main]
            if step.fallbacks:
                handlers.extend(step.fallbacks)
            self.fallback_manager.handlers = handlers
            result = await self.fallback_manager.execute([result])
        return result
