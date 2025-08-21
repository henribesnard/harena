"""Lightweight response generation utilities.

This module provides a minimal asynchronous generator used by the websocket
endpoint to stream back responses. In the real application this would bridge
with the more advanced ``ResponseGeneratorAgent``.
"""

from typing import AsyncGenerator


async def stream_response(message: str) -> AsyncGenerator[str, None]:
    """Yield a simple response for the provided message.

    The implementation is intentionally lightweight to avoid heavy dependencies
    during tests while illustrating how streaming would behave.
    """
    yield f"Response: {message}"
