import pytest
import time

from .harena_test_client import HarenaTestClient, REQUEST_TIMEOUT


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
