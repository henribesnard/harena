import pytest

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
