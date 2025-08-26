"""
Configuration globale des tests pour Harena - Compatible JWT user_service
"""
import os
import sys
import types
import pytest
import json
import asyncio
from collections import OrderedDict
from typing import Any
from pathlib import Path

# ============================================================================
# CONFIGURATION CRITIQUE SECRET_KEY - COMPATIBLE user_service
# ============================================================================

# Utilise la même clé que test_jwt_compatibility.py pour cohérence totale
SECRET_KEY_TEST = "a" * 32 + "b" * 32  # 64 caractères comme dans test_jwt_compatibility.py

# Configuration environment variables AVANT tous les imports
os.environ.setdefault("SECRET_KEY", SECRET_KEY_TEST)
os.environ.setdefault("JWT_ALGORITHM", "HS256")

# Configuration pour éviter les appels externes
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-deepseek-key-for-pytest")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key-for-pytest")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

# Configuration conversation service
os.environ.setdefault("CONVERSATION_SERVICE_ENABLED", "true")
os.environ.setdefault("CONVERSATION_SERVICE_DEBUG", "true")

# ============================================================================
# CONFIGURATION PYTEST
# ============================================================================

def pytest_configure():
    """Configuration exécutée avant tous les tests"""
    # Force la cohérence de SECRET_KEY
    if os.environ.get("SECRET_KEY") != SECRET_KEY_TEST:
        os.environ["SECRET_KEY"] = SECRET_KEY_TEST
        print(f"[PYTEST] SECRET_KEY forcée à: {SECRET_KEY_TEST[:10]}...")
    
    print(f"[PYTEST] Configuration test initialisée")
    print(f"[PYTEST] SECRET_KEY: {os.environ['SECRET_KEY'][:10]}...{os.environ['SECRET_KEY'][-10:]}")
    print(f"[PYTEST] JWT_ALGORITHM: {os.environ['JWT_ALGORITHM']}")
    print(f"[PYTEST] ENVIRONMENT: {os.environ['ENVIRONMENT']}")

# ============================================================================
# GESTION AUTOMATIQUE LOOP ASYNCIO
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Event loop pour les tests asynchrones"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def ensure_secret_key():
    """S'assure que SECRET_KEY est cohérente pour chaque test"""
    if os.environ.get("SECRET_KEY") != SECRET_KEY_TEST:
        os.environ["SECRET_KEY"] = SECRET_KEY_TEST

# ============================================================================
# STUBS POUR DÉPENDANCES EXTERNES (améliorés)
# ============================================================================

# Assure-toi que le stub autogen_agentchat est disponible
try:
    import autogen_agentchat
except ModuleNotFoundError:
    stub_dir = Path(__file__).resolve().parents[1] / "autogen_agentchat"
    if stub_dir.exists():
        sys.path.insert(0, str(stub_dir))

class GlobalSettings:
    """Settings stub avec configuration test"""
    OPENAI_API_KEY: str = "test-openai-key"
    SECRET_KEY: str = SECRET_KEY_TEST
    JWT_ALGORITHM: str = "HS256"
    ENVIRONMENT: str = "test"

settings = GlobalSettings()

def create_pydantic_stub():
    """Stub Pydantic amélioré pour les tests"""
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
    """Stub OpenAI pour tests"""
    openai_stub = types.ModuleType("openai")
    
    class _AsyncOpenAI:
        def __init__(self, *args, **kwargs):
            pass
    
    openai_stub.AsyncOpenAI = _AsyncOpenAI
    
    # Sous-modules types
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
    """Stub HTTPX pour tests"""
    httpx_stub = types.ModuleType("httpx")
    
    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        
        async def post(self, *args, **kwargs):
            return types.SimpleNamespace(
                json=lambda: {},
                raise_for_status=lambda: None,
                status_code=200
            )
        
        async def get(self, *args, **kwargs):
            return types.SimpleNamespace(
                json=lambda: {},
                raise_for_status=lambda: None,
                status_code=200
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
    """Stub AutoGen pour tests"""
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

def create_pydantic_settings_stub():
    """Stub pydantic_settings pour tests"""
    module = types.ModuleType("pydantic_settings")

    class _BaseSettings:
        def __init__(self, **values: Any):
            for k, v in values.items():
                setattr(self, k, v)

    module.BaseSettings = _BaseSettings
    module.SettingsConfigDict = dict
    return module

def create_aiohttp_stub():
    """Stub aiohttp pour tests"""
    aiohttp_stub = types.ModuleType("aiohttp")

    class _ClientSession:
        async def post(self, *args, **kwargs):
            return types.SimpleNamespace(status=200, json=lambda: {}, text=lambda: "")

        async def get(self, *args, **kwargs):
            return types.SimpleNamespace(status=200, json=lambda: {}, text=lambda: "")

        async def close(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

    class _ClientTimeout:
        def __init__(self, *args, **kwargs):
            pass

    class _TCPConnector:
        def __init__(self, *args, **kwargs):
            pass

    class _ClientError(Exception):
        pass

    aiohttp_stub.ClientSession = _ClientSession
    aiohttp_stub.ClientTimeout = _ClientTimeout
    aiohttp_stub.TCPConnector = _TCPConnector
    aiohttp_stub.ClientError = _ClientError
    return aiohttp_stub

def install_stubs():
    """Installation automatique des stubs nécessaires"""
    
    # Pydantic - utiliser la vraie si disponible
    try:
        import pydantic
        if not hasattr(pydantic, 'create_model'):
            def _create_model(name, **fields):
                return type(name, (pydantic.BaseModel,), fields)
            pydantic.create_model = _create_model
    except ImportError:
        sys.modules["pydantic"] = create_pydantic_stub()

    # pydantic_settings
    try:
        import pydantic_settings
    except ImportError:
        sys.modules["pydantic_settings"] = create_pydantic_settings_stub()
    
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

    # aiohttp
    try:
        import aiohttp
    except ImportError:
        sys.modules["aiohttp"] = create_aiohttp_stub()
    
    # Autogen
    try:
        import autogen
    except ImportError:
        sys.modules["autogen"] = create_autogen_stub()

# Installation automatique
install_stubs()

# Recharger la configuration pour prendre en compte les variables d'environnement
try:
    import importlib
    import config_service.config as config
    importlib.reload(config)
except ImportError:
    pass  # config_service peut ne pas être disponible dans tous les contextes

# ============================================================================
# FIXTURES DE TEST AVANCÉES
# ============================================================================

class DummyOpenAIClient:
    """Client OpenAI stub avec contenu préconfiguré"""

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
    """Configuration OpenAI centralisée pour les tests"""
    settings.OPENAI_API_KEY = settings.OPENAI_API_KEY or "test-key"
    return settings

@pytest.fixture
def openai_mock(openai_settings: GlobalSettings, monkeypatch: pytest.MonkeyPatch):
    """Mock OpenAI client avec contenu déterministe"""
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
    """Cache LRU en mémoire pour les tests"""
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

# ============================================================================
# FIXTURES JWT POUR TESTS D'AUTHENTIFICATION
# ============================================================================

@pytest.fixture
def test_secret_key():
    """Secret key pour les tests"""
    return SECRET_KEY_TEST

@pytest.fixture
def jwt_algorithm():
    """Algorithme JWT pour les tests"""
    return "HS256"

def reset_stubs():
    """Réinitialisation des stubs (utile pour les tests)"""
    modules_to_remove = [
        "pydantic", "openai", "openai.types", "openai.types.chat",
        "httpx", "autogen", "pydantic_settings", "aiohttp"
    ]
    for module in modules_to_remove:
        if module in sys.modules:
            del sys.modules[module]

    install_stubs()

# ============================================================================
# UTILITAIRES DE DEBUG POUR LES TESTS
# ============================================================================

def debug_test_environment():
    """Affiche l'environnement de test pour débugger"""
    print("=== DEBUG TEST ENVIRONMENT ===")
    print(f"SECRET_KEY: {os.environ.get('SECRET_KEY', 'NOT_SET')[:10]}...")
    print(f"JWT_ALGORITHM: {os.environ.get('JWT_ALGORITHM', 'NOT_SET')}")
    print(f"ENVIRONMENT: {os.environ.get('ENVIRONMENT', 'NOT_SET')}")
    print(f"DEEPSEEK_API_KEY: {'SET' if os.environ.get('DEEPSEEK_API_KEY') else 'NOT_SET'}")
    print("=" * 32)

# Appel automatique du debug si PYTEST_DEBUG=true
if os.environ.get("PYTEST_DEBUG", "").lower() == "true":
    debug_test_environment()

# ============================================================================
# VALIDATION FINALE
# ============================================================================

# Vérification finale de cohérence
assert os.environ["SECRET_KEY"] == SECRET_KEY_TEST, f"SECRET_KEY incohérente: {os.environ['SECRET_KEY'][:10]} != {SECRET_KEY_TEST[:10]}"
assert len(os.environ["SECRET_KEY"]) == 64, f"SECRET_KEY doit faire 64 caractères, actuel: {len(os.environ['SECRET_KEY'])}"

print(f"[CONFTEST] Configuration test validée - SECRET_KEY: {len(os.environ['SECRET_KEY'])} chars")