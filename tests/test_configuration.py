import importlib
import pytest


def reload_config():
    """Reload configuration module after environment changes."""
    import config_service.config as config
    importlib.reload(config)
    return config


def test_openai_config_from_env(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    monkeypatch.setenv("OPENAI_API_KEY", "key123")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://test")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-test")
    config = reload_config()
    settings = config.GlobalSettings()
    cfg = settings.get_openai_config()
    assert cfg["api_key"] == "key123"
    assert cfg["chat_model"] == "gpt-test"
    assert cfg["base_url"] == "http://test"


def test_model_cost_calculation():
    from monitoring import performance
    importlib.reload(performance)
    performance.record_openai_cost(1.5)
    assert performance._total_openai_cost == pytest.approx(1.5)
    performance.record_openai_cost(0.5)
    assert performance._total_openai_cost == pytest.approx(2.0)


def test_agent_specific_config(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    monkeypatch.setenv("OPENAI_API_KEY", "key123")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://test")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-base")
    monkeypatch.setenv("OPENAI_INTENT_MAX_TOKENS", "99")
    monkeypatch.setenv("OPENAI_INTENT_TEMPERATURE", "0.33")
    monkeypatch.setenv("OPENAI_INTENT_TOP_P", "0.77")
    monkeypatch.setenv("OPENAI_INTENT_TIMEOUT", "15")
    config = reload_config()
    settings = config.GlobalSettings()
    cfg = settings.get_openai_config("intent")
    assert cfg["max_tokens"] == 99
    assert cfg["temperature"] == 0.33
    assert cfg["top_p"] == 0.77
    assert cfg["timeout"] == 15


def test_fallback_configuration(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    monkeypatch.setenv("OPENAI_API_KEY", "key123")
    config = reload_config()
    settings = config.GlobalSettings()
    default_cfg = settings.get_openai_config()
    unknown_cfg = settings.get_openai_config("unknown")
    assert unknown_cfg == default_cfg


def test_invalid_config_handling(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    config = reload_config()
    with pytest.raises(ValueError):
        config.GlobalSettings(REDIS_URL="")
