import asyncio
import sys
import types

# Stub FastAPI dependencies for isolated unit tests
fastapi_module = types.ModuleType("fastapi")

class APIRouter:
    def __init__(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def post(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def include_router(self, *args, **kwargs):
        pass


def Depends(dep):
    return dep


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail


status = types.SimpleNamespace(
    HTTP_403_FORBIDDEN=403,
    HTTP_503_SERVICE_UNAVAILABLE=503,
    HTTP_500_INTERNAL_SERVER_ERROR=500,
)


class BackgroundTasks:
    def add_task(self, *args, **kwargs):
        pass


fastapi_module.APIRouter = APIRouter
fastapi_module.Depends = Depends
fastapi_module.HTTPException = HTTPException
fastapi_module.status = status
fastapi_module.BackgroundTasks = BackgroundTasks
class Request:
    pass
fastapi_module.Request = Request
sys.modules["fastapi"] = fastapi_module

responses_module = types.ModuleType("fastapi.responses")
class JSONResponse:
    pass
responses_module.JSONResponse = JSONResponse
sys.modules["fastapi.responses"] = responses_module

security_module = types.ModuleType("fastapi.security")
class HTTPBearer:
    def __init__(self, *args, **kwargs):
        pass
class HTTPAuthorizationCredentials:
    pass
security_module.HTTPBearer = HTTPBearer
security_module.HTTPAuthorizationCredentials = HTTPAuthorizationCredentials
sys.modules["fastapi.security"] = security_module

from conversation_service.api.routes import get_metrics
from conversation_service.utils.metrics import MetricsCollector


class DummyTeamManager:
    def get_team_performance(self):
        return {}


def test_metrics_endpoint_returns_json():
    metrics = MetricsCollector()
    user = {"permissions": ["view_metrics"]}
    result = asyncio.run(get_metrics(metrics=metrics, team_manager=DummyTeamManager(), user=user))
    assert "service_metrics" in result
    assert "system_info" in result
    assert "memory_usage" in result["system_info"]
    assert "cpu_usage" in result["system_info"]


if __name__ == "__main__":
    import pytest, sys

    sys.exit(pytest.main([__file__]))
