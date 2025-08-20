from pydantic_settings import BaseSettings, SettingsConfigDict


class AutoGenSettings(BaseSettings):
    """Paramètres généraux et spécifiques à AutoGen."""

    ENVIRONMENT: str = "development"
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8080"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    SECRET_KEY: str = ""
    DATABASE_URL: str = ""
    REDIS_URL: str = ""
    REDISCLOUD_URL: str = ""
    REDIS_PASSWORD: str | None = None

    BONSAI_URL: str = ""
    BRIDGE_CLIENT_ID: str = ""
    BRIDGE_CLIENT_SECRET: str = ""
    DEEPSEEK_API_KEY: str = ""

    WORKFLOW_TIMEOUT_SECONDS: int = 45
    HEALTH_CHECK_INTERVAL_SECONDS: int = 300
    AUTO_RECOVERY_ENABLED: bool = True
    INITIAL_HEALTH_CHECK_DELAY_SECONDS: int = 60
    INITIAL_HEALTH_CHECK: bool = False
    AGENT_FAILURE_THRESHOLD: int = 3
    ORCHESTRATOR_PERFORMANCE_THRESHOLD_MS: int = 30000
    AGENT_REACTIVATION_COOLDOWN_SECONDS: int = 60

    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_TIMEOUT: int = 30
    SEARCH_SERVICE_URL: str = "http://localhost:8000/api/v1/search"
    MAX_CONVERSATION_HISTORY: int = 100

    MEMORY_CACHE_SIZE: int = 2000
    MEMORY_CACHE_TTL: int = 300
    CACHE_TTL: int = 3600
    CACHE_TTL_RESPONSE: int = 60
    REDIS_CACHE_PREFIX: str = "harena_conv"
    REDIS_MAX_CONNECTIONS: int = 10
    REDIS_CACHE_ENABLED: bool = True

    SEARCH_RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    SEARCH_RATE_LIMIT_WINDOW_SECONDS: int = 60
    SEARCH_CACHE_TTL: int = 30

    INTENT_TIMEOUT_SECONDS: float = 10
    INTENT_MAX_RETRIES: int = 3
    INTENT_BACKOFF_BASE: float = 1
    SEARCH_QUERY_DEFAULT_LIMIT: int = 100

    PERFORMANCE_ALERT_THRESHOLD_MS: float = 1000.0
    ERROR_RATE_ALERT_THRESHOLD: float = 0.05
    METRICS_COLLECTION_INTERVAL: int = 60
    ENABLE_METRICS: bool = True
    VALIDATION_STRICT: bool = True
    CS_AUTO_IMPORT_SUBPACKAGES: int = 0

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

