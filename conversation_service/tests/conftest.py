"""
Module de stubs pour les tests unitaires.
Fournit des implémentations minimales des dépendances externes
pour permettre l'exécution des tests sans les vraies dépendances.
"""
import sys
import types
import pytest
import json
from collections import OrderedDict
from typing import Any
# Minimal settings stub to avoid external dependency

class GlobalSettings:
    OPENAI_API_KEY: str = "test-key"


settings = GlobalSettings()


def create_pydantic_stub():
    """Crée un stub minimal pour pydantic."""
    pydantic_stub = types.ModuleType("pydantic")
    
    class _BaseModel:
        def __init__(self, **data):
            for k, v in data.items():
                setattr(self, k, v)
        
        def dict(self, *args, **kwargs):
            result = {}
            for k, v in self.__dict__.items():
                if hasattr(v, "dict"):
                    result[k] = v.dict()
                else:
                    result[k] = v
            return result
        
        def model_dump(self, *args, **kwargs):
            """Alias pour dict() dans Pydantic v2"""
            return self.dict(*args, **kwargs)
    
    def _Field(*args, **kwargs):
        return kwargs.get("default")
    
    def _field_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def _model_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def _create_model(name, **fields):
        """Stub pour create_model"""
        return type(name, (_BaseModel,), fields)

    pydantic_stub.BaseModel = _BaseModel
    pydantic_stub.Field = _Field
    pydantic_stub.field_validator = _field_validator
    pydantic_stub.model_validator = _model_validator
    pydantic_stub.create_model = _create_model
    pydantic_stub.ConfigDict = dict
    pydantic_stub.ValidationError = type("ValidationError", (Exception,), {})
    
    return pydantic_stub


def create_openai_stub():
    """Crée un stub minimal pour openai."""
    openai_stub = types.ModuleType("openai")
    
    class _AsyncOpenAI:
        def __init__(self, *args, **kwargs):
            pass
    
    openai_stub.AsyncOpenAI = _AsyncOpenAI
    
    # Créer les sous-modules types
    openai_types = types.ModuleType("openai.types")
    openai_chat = types.ModuleType("openai.types.chat")
    openai_chat.ChatCompletion = object
    openai_types.chat = openai_chat
    openai_stub.types = openai_types
    
    # Enregistrer les sous-modules
    sys.modules["openai.types"] = openai_types
    sys.modules["openai.types.chat"] = openai_chat
    
    return openai_stub


def create_httpx_stub():
    """Crée un stub minimal pour httpx."""
    httpx_stub = types.ModuleType("httpx")
    
    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        
        async def post(self, *args, **kwargs):
            return types.SimpleNamespace(
                json=lambda: {},
                raise_for_status=lambda: None
            )
        
        async def get(self, *args, **kwargs):
            return types.SimpleNamespace(
                json=lambda: {},
                raise_for_status=lambda: None
            )
        
        async def aclose(self):
            pass
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, *args):
            await self.aclose()
    
    httpx_stub.AsyncClient = _AsyncClient
    httpx_stub.RequestError = type("RequestError", (Exception,), {})
    httpx_stub.HTTPStatusError = type("HTTPStatusError", (Exception,), {})
    
    return httpx_stub


def create_autogen_stub():
    """Crée un stub minimal pour autogen."""
    autogen_stub = types.ModuleType("autogen")
    
    class _AssistantAgent:
        def __init__(self, name=None, system_message=None, llm_config=None, 
                     max_consecutive_auto_reply=None, description=None, **kwargs):
            self.name = name
            self.system_message = system_message
            self.llm_config = llm_config
            self.max_consecutive_auto_reply = max_consecutive_auto_reply
            self.description = description
    
    autogen_stub.AssistantAgent = _AssistantAgent
    
    return autogen_stub


def install_stubs():
    """Installe tous les stubs nécessaires."""
    
    # Pydantic - utiliser la vraie bibliothèque si disponible
    try:
        import pydantic
        if not hasattr(pydantic, 'create_model'):
            # Ajouter create_model si manquant
            def _create_model(name, **fields):
                return type(name, (pydantic.BaseModel,), fields)
            pydantic.create_model = _create_model
    except ImportError:
        sys.modules["pydantic"] = create_pydantic_stub()
    
    # OpenAI
    try:
        import openai
    except ImportError:
        sys.modules["openai"] = create_openai_stub()
    
    # HTTPX
    try:
        import httpx
    except ImportError:
        sys.modules["httpx"] = create_httpx_stub()
    
    # Autogen
    try:
        import autogen
    except ImportError:
        sys.modules["autogen"] = create_autogen_stub()


# Installer les stubs au chargement du module
install_stubs()


# Pour les tests, exposer une fonction de réinitialisation
def reset_stubs():
    """Réinitialise tous les stubs (utile pour les tests)."""
    modules_to_remove = [
        "pydantic", "openai", "openai.types", "openai.types.chat",
        "httpx", "autogen"
    ]
    for module in modules_to_remove:
        if module in sys.modules:
            del sys.modules[module]

    install_stubs()


# Fixtures for tests
class DummyOpenAIClient:
    """Simple OpenAI client stub returning preconfigured content."""

    def __init__(self, content: str):
        self._content = content

        class _Completions:
            async def create(_self, *args, **kwargs):
                choice = types.SimpleNamespace(
                    message=types.SimpleNamespace(content=content)
                )
                return types.SimpleNamespace(choices=[choice])

        class _Chat:
            def __init__(self):
                self.completions = _Completions()

        self.chat = _Chat()


@pytest.fixture(scope="session")
def openai_settings() -> GlobalSettings:
    """Centralized OpenAI configuration for tests."""
    settings.OPENAI_API_KEY = settings.OPENAI_API_KEY or "test-key"
    return settings


@pytest.fixture
def openai_mock(openai_settings: GlobalSettings, monkeypatch: pytest.MonkeyPatch):
    """Provide a mock OpenAI client with deterministic content."""
    monkeypatch.setenv("OPENAI_API_KEY", openai_settings.OPENAI_API_KEY)
    payload = {
        "intent_type": "GREETING",
        "intent_category": "GREETING",
        "confidence": 0.9,
        "entities": [],
    }
    return DummyOpenAIClient(json.dumps(payload))


@pytest.fixture
def cache():
    """Provide an in-memory LRU cache for tests."""
    class LRUCache:
        def __init__(self, maxsize: int = 128):
            self.maxsize = maxsize
            self._cache: OrderedDict[str, Any] = OrderedDict()

        def get(self, key: str) -> Any:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

        def set(self, key: str, value: Any) -> None:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    return LRUCache(maxsize=16)
