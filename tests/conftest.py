import sys
import types


# Minimal stub for pydantic to avoid external dependency during tests
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

pydantic_stub.BaseModel = _BaseModel
pydantic_stub.Field = _Field
pydantic_stub.field_validator = _field_validator
pydantic_stub.model_validator = _model_validator
pydantic_stub.ValidationError = type("ValidationError", (Exception,), {})
sys.modules.setdefault("pydantic", pydantic_stub)


# Stub for openai module used by DeepSeekClient
openai_stub = types.ModuleType("openai")

class _AsyncOpenAI:
    pass

openai_stub.AsyncOpenAI = _AsyncOpenAI

openai_types = types.ModuleType("openai.types")
openai_chat = types.ModuleType("openai.types.chat")
openai_chat.ChatCompletion = object
openai_types.chat = openai_chat
openai_stub.types = openai_types

sys.modules.setdefault("openai", openai_stub)
sys.modules.setdefault("openai.types", openai_types)
sys.modules.setdefault("openai.types.chat", openai_chat)


# Stub for httpx library
httpx_stub = types.ModuleType("httpx")

class _AsyncClient:
    def __init__(self, *args, **kwargs):
        pass

    async def post(self, *args, **kwargs):
        return types.SimpleNamespace(json=lambda: {}, raise_for_status=lambda: None)

    async def aclose(self):
        pass

httpx_stub.AsyncClient = _AsyncClient
httpx_stub.RequestError = Exception
httpx_stub.HTTPStatusError = Exception
sys.modules.setdefault("httpx", httpx_stub)


# Stub for autogen AssistantAgent
autogen_stub = types.ModuleType("autogen")

class _AssistantAgent:
    def __init__(self, name=None, system_message=None, llm_config=None, max_consecutive_auto_reply=None, description=None, **kwargs):
        self.name = name
        self.system_message = system_message
        self.llm_config = llm_config
        self.max_consecutive_auto_reply = max_consecutive_auto_reply
        self.description = description

autogen_stub.AssistantAgent = _AssistantAgent
sys.modules.setdefault("autogen", autogen_stub)

