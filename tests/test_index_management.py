import pytest
from enrichment_service.storage.index_management import (
    ensure_template_and_policy,
    INDEX_TEMPLATE,
    ILM_POLICY,
)


class MockResponse:
    def __init__(self, status=200, text=""):
        self.status = status
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def text(self):
        return self._text


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


class FailingILMSession(MockSession):
    def put(self, url, json):
        self.calls.append((url, json))
        if "_ilm" in url:
            return MockResponse(status=400, text="ILM unsupported")
        return MockResponse()


class ForbiddenTemplateSession(MockSession):
    def put(self, url, json):
        self.calls.append((url, json))
        if "_index_template" in url:
            return MockResponse(status=403, text="Forbidden")
        return MockResponse()


@pytest.mark.asyncio
async def test_ensure_template_and_policy_bypasses_ilm_on_error():
    session = FailingILMSession()
    await ensure_template_and_policy(session, "http://es:9200")

    # ILM policy call attempted first
    assert session.calls[0][0] == (
        "http://es:9200/_ilm/policy/harena_transactions_policy"
    )

    # Template call should not contain ILM settings
    tmpl_settings = session.calls[1][1]["template"]["settings"]
    assert "index.lifecycle.name" not in tmpl_settings
    assert "index.lifecycle.rollover_alias" not in tmpl_settings


@pytest.mark.asyncio
async def test_ensure_template_and_policy_continues_on_template_forbidden():
    session = ForbiddenTemplateSession()
    await ensure_template_and_policy(session, "http://es:9200")

    # Both ILM policy and template endpoints are called
    assert session.calls[0][0] == (
        "http://es:9200/_ilm/policy/harena_transactions_policy"
    )
    assert session.calls[1][0] == (
        "http://es:9200/_index_template/harena_transactions_template"
    )


@pytest.mark.asyncio
async def test_ensure_template_and_policy_skipped_when_disabled(monkeypatch):
    session = MockSession()
    monkeypatch.setattr(
        "enrichment_service.storage.index_management.settings.DISABLE_INDEX_TEMPLATE",
        True,
    )
    await ensure_template_and_policy(session, "http://es:9200")

    assert session.calls == []
