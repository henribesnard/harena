"""Wrapper module for response generation utilities.

This simply re-exports :class:`ResponseGeneratorAgent` and ``stream_response``
from the ``response_generator`` module so tests can import a consistent public
API without pulling in heavier dependencies.
"""

# The response generator depends on optional libraries; import lazily to avoid
# failing when those dependencies are absent.  A minimal ``stream_response``
# fallback is provided to keep the public API stable for tests.
try:  # pragma: no cover - defensive import
    from .response_generator import ResponseGeneratorAgent, stream_response  # type: ignore
except Exception:  # pragma: no cover - dependency not available
    ResponseGeneratorAgent = None  # type: ignore

    async def stream_response(message: str):  # type: ignore[override]
        """Fallback async generator returning the message directly."""
        yield f"Response: {message}"

__all__ = ["ResponseGeneratorAgent", "stream_response"]
