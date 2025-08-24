import pytest

from enrichment_service.core.processor import ElasticsearchTransactionProcessor


class HealthyESClient:
    async def ping(self):
        return True


class FailingESClient:
    async def ping(self):
        return False


class DummyAccountService:
    pass


@pytest.mark.asyncio
async def test_health_check_healthy():
    processor = ElasticsearchTransactionProcessor(HealthyESClient(), DummyAccountService())
    status = await processor.health_check()
    assert status["status"] == "healthy"
    assert status["elasticsearch"]["available"] is True
    assert status.get("database", {}).get("available") is True


@pytest.mark.asyncio
async def test_health_check_failing_es():
    processor = ElasticsearchTransactionProcessor(FailingESClient(), DummyAccountService())
    status = await processor.health_check()
    assert status["status"] != "healthy"
    assert status["elasticsearch"]["available"] is False
