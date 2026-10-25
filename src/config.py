import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys & Providers
    GEMINI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    
    # Models
    LLM_MODEL: str = "gemini-2.5-flash"
    EMBEDDING_MODEL: str = "models/text-embedding-004"
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DOCUMENTS_DIR: Path = DATA_DIR / "documents"
    SANCTIONS_FILE: Path = DATA_DIR / "sanctions_mock_db.json"
    CHROMA_PERSIST_DIR: Path = BASE_DIR / "chroma_db"
    
    # Chunking & RAG Configurations
    BASELINE_CHUNK_SIZE: int = 1000
    BASELINE_CHUNK_OVERLAP: int = 100
    OPTIMIZED_CHUNK_SIZE: int = 500
    OPTIMIZED_CHUNK_OVERLAP: int = 60
    TOP_K: int = 4
    
    # Cost tracking (USD per 1M tokens) - Gemini Flash baseline
    INPUT_COST_PER_1M: float = 0.075
    OUTPUT_COST_PER_1M: float = 0.300
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    @property
    def effective_api_key(self) -> str:
        return self.GEMINI_API_KEY or self.GOOGLE_API_KEY or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")


settings = Settings()
