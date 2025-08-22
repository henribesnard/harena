import sys
import types
import pytest
import asyncio

# Stub external dependencies to allow importing the module without optional packages.

redis_asyncio = types.ModuleType("redis.asyncio")

async def _noop(*args, **kwargs):
    return None

redis_asyncio.from_url = lambda *args, **kwargs: types.SimpleNamespace(
    get=_noop, set=_noop, delete=_noop, close=_noop
)
redis_exceptions = types.ModuleType("redis.exceptions")
redis_exceptions.RedisError = Exception
redis_module = types.ModuleType("redis")
redis_module.asyncio = redis_asyncio
redis_module.exceptions = redis_exceptions
sys.modules.setdefault("redis.asyncio", redis_asyncio)
sys.modules.setdefault("redis.exceptions", redis_exceptions)
sys.modules.setdefault("redis", redis_module)

aiohttp_module = types.ModuleType("aiohttp")
aiohttp_module.ClientSession = object
aiohttp_module.ClientTimeout = lambda *args, **kwargs: None
aiohttp_module.ClientError = Exception
sys.modules.setdefault("aiohttp", aiohttp_module)

openai_module = types.ModuleType("openai")
openai_module.AsyncOpenAI = object
sys.modules.setdefault("openai", openai_module)

fastapi_module = types.ModuleType("fastapi")
class HTTPException(Exception):
    pass
class WebSocket:
    query_params = {}
    async def close(self, code: int) -> None:
        return None
status = types.SimpleNamespace(WS_1008_POLICY_VIOLATION=1008, HTTP_403_FORBIDDEN=403)
fastapi_module.HTTPException = HTTPException
fastapi_module.WebSocket = WebSocket
fastapi_module.status = status
sys.modules.setdefault("fastapi", fastapi_module)

# Stub configuration modules
config_pkg = types.ModuleType("config")
config_pkg.__path__ = []  # mark as package
autogen_mod = types.ModuleType("config.autogen_config")
class AutogenConfig:  # simple placeholder
    pass
autogen_settings = AutogenConfig()
autogen_mod.AutogenConfig = AutogenConfig
autogen_mod.autogen_settings = autogen_settings
config_pkg.autogen_config = autogen_mod
sys.modules.setdefault("config", config_pkg)
sys.modules.setdefault("config.autogen_config", autogen_mod)

config_service_pkg = types.ModuleType("config_service")
config_service_pkg.__path__ = []
config_service_config_mod = types.ModuleType("config_service.config")
settings = types.SimpleNamespace(
    REDIS_URL="redis://localhost", REDIS_CACHE_PREFIX="test", SEARCHBOX_URL="", ELASTICSEARCH_URL="http://localhost"
)
config_service_config_mod.settings = settings
config_service_pkg.config = config_service_config_mod
sys.modules.setdefault("config_service", config_service_pkg)
sys.modules.setdefault("config_service.config", config_service_config_mod)

openai_config_mod = types.ModuleType("openai_config")
openai_config_mod.openai_config = types.SimpleNamespace(api_key="", base_url="")
sys.modules.setdefault("openai_config", openai_config_mod)

from conversation_service.api import dependencies


def test_get_cache_client_returns_instance():
    client = asyncio.run(dependencies.get_cache_client())
    assert isinstance(client, dependencies.CacheClient)
