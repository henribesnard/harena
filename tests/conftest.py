import sys
import types
import pytest


@pytest.fixture
def httpx_stub(monkeypatch):
    httpx = types.ModuleType("httpx")

    class Response:
        def __init__(self, *args, **kwargs):
            self._json = kwargs.get("json", {})

        def json(self):
            return self._json

        def raise_for_status(self):
            pass

    class AsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def post(self, *args, **kwargs):
            return Response()

        async def aclose(self):
            pass

    class HTTPStatusError(Exception):
        pass

    class TimeoutException(Exception):
        pass

    class RequestError(Exception):
        pass

    httpx.Response = Response
    httpx.AsyncClient = AsyncClient
    httpx.HTTPStatusError = HTTPStatusError
    httpx.TimeoutException = TimeoutException
    httpx.RequestError = RequestError

    monkeypatch.setitem(sys.modules, "httpx", httpx)
    yield httpx

