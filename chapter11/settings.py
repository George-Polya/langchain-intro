import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

CHAPTER_DIR = Path(__file__).resolve().parent
ROOT_DIR = CHAPTER_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str = ""
    TAVILY_API_KEY: str
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "agent-book"

    # for Application
    openai_smart_model: str = "gpt-5.4-nano"
    openai_embedding_model: str = "text-embedding-3-small"
    anthropic_smart_model: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.0
    default_reflection_db_path: str = "tmp/reflection_db.json"

    def __init__(self, **values):
        super().__init__(**values)
        self._set_env_variables()

    def _set_env_variables(self):
        for key in self.__annotations__.keys():
            if key.isupper():
                os.environ[key] = getattr(self, key)
