import os
from pathlib import Path

from settings import Settings, settings
from openai_config import OpenAIConfig
from autogen_config import AutoGenConfig


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
