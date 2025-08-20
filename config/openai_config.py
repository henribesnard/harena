from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAISettings(BaseSettings):
    """Paramètres de configuration pour l'API OpenAI."""

    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_REASONER_MODEL: str = "gpt-4o-mini"
    OPENAI_MAX_TOKENS: int = 2048
    OPENAI_TEMPERATURE: float = 1.0
    OPENAI_TOP_P: float = 0.95
    OPENAI_TIMEOUT: int = 30
    OPENAI_MAX_PROMPT_CHARS: int = 2000

    OPENAI_INTENT_MAX_TOKENS: int = 80
    OPENAI_INTENT_TEMPERATURE: float = 0.1
    OPENAI_INTENT_TIMEOUT: int = 6
    OPENAI_INTENT_TOP_P: float = 0.9

    OPENAI_ENTITY_MAX_TOKENS: int = 60
    OPENAI_ENTITY_TEMPERATURE: float = 0.05
    OPENAI_ENTITY_TIMEOUT: int = 5
    OPENAI_ENTITY_TOP_P: float = 0.8

    OPENAI_QUERY_MAX_TOKENS: int = 200
    OPENAI_QUERY_TEMPERATURE: float = 0.2
    OPENAI_QUERY_TIMEOUT: int = 8
    OPENAI_QUERY_TOP_P: float = 0.9

    OPENAI_RESPONSE_MAX_TOKENS: int = 300
    OPENAI_RESPONSE_TEMPERATURE: float = 0.7
    OPENAI_RESPONSE_TIMEOUT: int = 12
    OPENAI_RESPONSE_TOP_P: float = 0.95

    OPENAI_EXPECTED_LATENCY_MS: int = 1500

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

