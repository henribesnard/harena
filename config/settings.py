from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_TITLE: str = "Harena API"
    APP_VERSION: str = "0.1.0"


settings = Settings()
