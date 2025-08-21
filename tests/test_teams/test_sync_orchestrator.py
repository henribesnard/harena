import sys
import types
from dataclasses import dataclass

import pytest

sys.modules.setdefault("pydantic_settings", types.SimpleNamespace(BaseSettings=object, SettingsConfigDict=dict))
sys.modules.setdefault("config_service.config", types.SimpleNamespace(settings=types.SimpleNamespace()))

from sync_service.sync_manager import orchestrator


@dataclass
class MockSyncItem:
    user_id: int
    bridge_item_id: int
    status: int
    status_code_info: str | None = None
    status_description: str | None = None
    provider_id: int | None = None
    account_types: str | None = None
    needs_user_action: bool = False
    last_successful_refresh: any = None
    last_try_refresh: any = None
    id: int = 1

# provide class attribute for query comparisons
MockSyncItem.bridge_item_id = 0


class FakeQuery:
    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return None


class FakeSession:
    def query(self, model):
        return FakeQuery()

    def add(self, obj):
        self.obj = obj

    def commit(self):
        pass

    def refresh(self, obj):
        pass

    def rollback(self):
        pass


@pytest.mark.asyncio
async def test_create_or_update_sync_item_sets_needs_user_action(monkeypatch):
    monkeypatch.setattr(orchestrator, "SyncItem", MockSyncItem)
    monkeypatch.setattr(orchestrator, "get_contextual_logger", lambda *a, **k: orchestrator.logging.getLogger("test"))

    db = FakeSession()
    item = await orchestrator.create_or_update_sync_item(
        db, user_id=1, bridge_item_id=123, item_data={"status": 402}
    )
    assert item.needs_user_action is True
