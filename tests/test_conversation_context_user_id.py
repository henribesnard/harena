import sys
from pathlib import Path
import types

# Ensure project root is on sys.path for module imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Minimal pydantic stub for tests
class _BaseModel:
    def __init__(self, **data):
        for k, v in data.items():
            setattr(self, k, v)

    def dict(self):
        result = {}
        for k, v in self.__dict__.items():
            if hasattr(v, "dict"):
                result[k] = v.dict()
            elif isinstance(v, list):
                result[k] = [item.dict() if hasattr(item, "dict") else item for item in v]
            else:
                result[k] = v
        return result


pydantic_stub = types.SimpleNamespace(
    BaseModel=_BaseModel,
    Field=lambda *args, **kwargs: None,
    field_validator=lambda *args, **kwargs: (lambda f: f),
    model_validator=lambda *args, **kwargs: (lambda f: f),
    ValidationError=Exception,
)
sys.modules.setdefault("pydantic", pydantic_stub)

from conversation_service.core.conversation_manager import ConversationManager


def test_add_turn_sets_user_id_in_context():
    async def run_test():
        manager = ConversationManager()
        await manager.add_turn(
            conversation_id="conv1",
            user_id=123,
            user_msg="hello",
            assistant_msg="hi",
        )
        context = await manager.get_context("conv1")
        assert context.user_id == 123

    import asyncio

    asyncio.run(run_test())
