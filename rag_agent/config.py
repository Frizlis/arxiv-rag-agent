import os
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Load and validate application configuration."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8-sig",
        case_sensitive=False,
        extra="forbid",
    )

    llm_provider: Literal["ollama"]
    llm_model: str = Field(min_length=1)
    ollama_base_url: AnyHttpUrl
    llm_temperature: float = Field(ge=0)
    llm_context_length: int = Field(gt=0)
    llm_thinking: bool

    embedding_provider: Literal["sentence_transformers"]
    embedding_model: str = Field(min_length=1)
    embedding_revision: str = Field(min_length=1)
    embedding_device: Literal["cpu", "cuda"]
    embedding_dim: int = Field(gt=0)
    embedding_local_files_only: bool

    local_storage_root: Path
    corpus_path: Path
    questions_path: Path

    @field_validator(
        "local_storage_root",
        "corpus_path",
        "questions_path",
        mode="after",
    )
    @classmethod
    def resolve_project_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            value = PROJECT_ROOT / value
        return value.resolve()

    @field_validator("local_storage_root", mode="after")
    @classmethod
    def validate_storage_drive(cls, value: Path) -> Path:
        if os.name == "nt" and value.drive.casefold() != "d:":
            raise ValueError("Local storage must be on drive D:")
        return value

    @property
    def embedding_cache_dir(self) -> Path:
        return self.local_storage_root / "huggingface" / "hub"
