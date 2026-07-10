from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str
    ANTHROPIC_API_KEY: str
    JWT_SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    CROSSREF_POLITE_EMAIL: str = "admin@example.com"
    ENV: str = "dev"
    VOYAGE_API_KEY: str = ""
    INTERNAL_SECRET_TOKEN: str = "CHANGE_ME_INSECURE_DEFAULT"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
