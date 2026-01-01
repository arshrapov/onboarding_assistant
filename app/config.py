"""
Configuration settings for the Onboarding Assistant application.
Uses pydantic-settings for environment variable management.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""



    # Data directories
    data_dir: str = "/app/data"
    repos_dir: str = "/app/data/repos"
    indexes_dir: str = "/app/data/indexes"
    cache_dir: str = "/app/data/cache"

    # File processing settings
    max_file_size_mb: int = 5
    supported_extensions: List[str] = [
        ".py", ".js", ".ts", ".tsx", ".jsx",
        ".java", ".go", ".rs", ".cpp", ".c",
        ".h", ".hpp", ".cs", ".rb", ".php",
        ".md", ".txt", ".json", ".yaml", ".yml"
    ]

    # Server settings for Gradio
    server_port: int = 7860
    server_host: str = "0.0.0.0"

    # RAG settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5

    # Required: Gemini API key
    gemini_api_key: str
    
    # LLM settings
    gemini_model: str = "gemini-1.5-flash"
    embedding_model: str = "models/embedding-001"
    max_tokens: int = 8192
    temperature: float = 0.7

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()