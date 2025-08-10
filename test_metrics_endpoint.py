import asyncio
import sys
import types

# Stub FastAPI components
fastapi_stub = types.ModuleType("fastapi")


class _APIRouter:
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
        return None


def _depends(dep=None):
    return dep


class _HTTPException(Exception):
    def __init__(self, status_code=None, detail=None, headers=None):
        self.status_code = status_code
        self.detail = detail
        self.headers = headers


class _BackgroundTasks:
    ...


fastapi_stub.APIRouter = _APIRouter
fastapi_stub.Depends = _depends
fastapi_stub.HTTPException = _HTTPException
fastapi_stub.status = types.SimpleNamespace(
    HTTP_503_SERVICE_UNAVAILABLE=503,
    HTTP_422_UNPROCESSABLE_ENTITY=422,
    HTTP_401_UNAUTHORIZED=401,
    HTTP_429_TOO_MANY_REQUESTS=429,
)
fastapi_stub.BackgroundTasks = _BackgroundTasks
fastapi_stub.Request = type("Request", (), {})

fastapi_responses = types.ModuleType("fastapi.responses")


class _JSONResponse(dict):
    def __init__(self, content=None, status_code=200):
        super().__init__(content or {})
        self.status_code = status_code


fastapi_responses.JSONResponse = _JSONResponse

fastapi_security = types.ModuleType("fastapi.security")
fastapi_security.HTTPBearer = type("HTTPBearer", (), {"__init__": lambda self, auto_error=False: None})
fastapi_security.HTTPAuthorizationCredentials = type(
    "HTTPAuthorizationCredentials", (), {"__init__": lambda self, scheme="", credentials="": None}
)

sys.modules.setdefault("fastapi", fastapi_stub)
sys.modules.setdefault("fastapi.responses", fastapi_responses)
sys.modules.setdefault("fastapi.security", fastapi_security)

# Minimal pydantic stub
pydantic_stub = types.ModuleType("pydantic")


class BaseModel:
    def __init__(self, **data):
        for k, v in data.items():
            setattr(self, k, v)

    def model_dump(self):
        return self.__dict__


def Field(default=None, *args, **kwargs):
    return default


def field_validator(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


def model_validator(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


pydantic_stub.BaseModel = BaseModel
pydantic_stub.Field = Field
pydantic_stub.field_validator = field_validator
pydantic_stub.model_validator = model_validator
pydantic_stub.ValidationError = type("ValidationError", (Exception,), {})
sys.modules["pydantic"] = pydantic_stub

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
    import pytest

    sys.exit(pytest.main([__file__]))
