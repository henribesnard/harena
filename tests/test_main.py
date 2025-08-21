import pytest
import time
import sys
import os

# Ajouter le répertoire parent au path pour permettre les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Minimal HTTP client used for testing timeout behaviour
REQUEST_TIMEOUT = 5


class HarenaTestClient:
    def __init__(self, base_url: str, session: object):
        self.base_url = base_url.rstrip("/")
        self.session = session
        self._token = ""
        self._token_expiry = 0.0
        self._refresh_token()

    def _refresh_token(self) -> None:
        self._token = "test-token"
        self._token_expiry = time.time() + 60

    def _ensure_token(self) -> None:
        if time.time() >= self._token_expiry - 5:
            self._refresh_token()

    def _make_request(self, method: str, endpoint: str, **kwargs):
        self._ensure_token()
        url = f"{self.base_url}{endpoint}"
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {self._token}"
        return self.session.request(method, url, timeout=REQUEST_TIMEOUT, headers=headers, **kwargs)

    def get(self, endpoint: str, **kwargs):
        return self._make_request("GET", endpoint, **kwargs)


class TimeoutException(Exception):
    pass


def test_timeout_triggers_error():
    class SlowSession:
        def request(self, method, url, timeout=None, **kwargs):
            assert timeout == REQUEST_TIMEOUT
            raise TimeoutException("timed out")

    client = HarenaTestClient("http://example.com", session=SlowSession())
    with pytest.raises(TimeoutException):
        client.get("/slow")


def test_authorization_and_refresh():
    """Ensure Authorization header is added and token refresh occurs."""

    class RecordingSession:
        def __init__(self):
            self.headers = None

        def request(self, method, url, timeout=None, **kwargs):
            self.headers = kwargs.get("headers")
            return {}

    session = RecordingSession()
    client = HarenaTestClient("http://example.com", session=session)
    client._token_expiry = time.time() - 1  # force refresh
    client.get("/any")
    assert session.headers["Authorization"] == "Bearer test-token"