"""
Application configuration
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
# parents[0]=core, [1]=app, [2]=backend  ✅

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ENV_PATH), env_file_encoding="utf-8")

    # API Settings
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "Multimodal Document Chat"

    # Database
    DATABASE_URL: str = "postgresql+psycopg2://docuser:docpass@localhost:5432/docdb"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4.1-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Upload Settings
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024

    # Vector Store
    EMBEDDING_DIMENSION: int = 1536
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 5

settings = Settings()
