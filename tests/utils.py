from typing import Any


def assert_status(response: Any, expected: int = 200) -> None:
    """Assert that a HTTP response has the expected status code."""
    assert response.status_code == expected, (
        f"expected {expected}, got {response.status_code}: {getattr(response, 'text', '')}"
    )


class SyncMock:
    """Simple synchronous mock callable recording received calls."""

    def __init__(self, return_value: Any = None):
        self.return_value = return_value
        self.calls = []

    def __call__(self, *args, **kwargs):  # pragma: no cover - trivial
        self.calls.append((args, kwargs))
        return self.return_value


class AsyncMock:
    """Simple asynchronous mock callable recording received calls."""

    def __init__(self, return_value: Any = None):
        self.return_value = return_value
        self.calls = []

    async def __call__(self, *args, **kwargs):  # pragma: no cover - trivial
        self.calls.append((args, kwargs))
        return self.return_value
