import os
from pathlib import Path

from config import (
    get_autogen_config,
    get_openai_config,
    get_settings,
)
from config.settings import Settings
from config.openai_config import OpenAIConfig
from config.autogen_config import AutoGenConfig


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "testing")
    cfg = Settings()
    assert cfg.environment == "testing"
    assert Path(cfg.base_dir).exists()


def test_openai_config_reads_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-test")
    cfg = OpenAIConfig()
    assert cfg.api_key == "sk-test"
    assert cfg.chat_model == "gpt-test"


def test_autogen_config_builds_params(monkeypatch):
    monkeypatch.setenv("AUTOGEN_AGENT_NAME", "unit-agent")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test2")
    auto_cfg = AutoGenConfig()
    params = auto_cfg.build_agent_params("hello")
    assert params.name == "unit-agent"
    assert params.model_client_config["api_key"] == "sk-test2"
    assert params.system_message == "hello"


def test_get_helpers_use_singletons():
    assert get_settings() is get_settings()
    assert get_openai_config() is get_openai_config()
    assert get_autogen_config() is get_autogen_config()
