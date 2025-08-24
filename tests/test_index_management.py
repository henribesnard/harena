import pytest
from enrichment_service.storage.index_management import (
    ensure_template_and_policy,
    INDEX_TEMPLATE,
    ILM_POLICY,
)


class MockResponse:
    def __init__(self, status=200):
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def text(self):
        return ""


class MockSession:
    def __init__(self):
        self.calls = []

    def put(self, url, json):
        self.calls.append((url, json))
        return MockResponse()


@pytest.mark.asyncio
async def test_ensure_template_and_policy_calls_es():
    session = MockSession()
    await ensure_template_and_policy(session, "http://es:9200")

    assert session.calls[0] == (
        "http://es:9200/_ilm/policy/harena_transactions_policy",
        ILM_POLICY,
    )
    assert session.calls[1] == (
        "http://es:9200/_index_template/harena_transactions_template",
        INDEX_TEMPLATE,
    )
