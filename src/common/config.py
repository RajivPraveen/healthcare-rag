"""Central configuration, loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM providers -----------------------------------------------------
    llm_provider: str = "auto"

    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5:7b-instruct"

    judge_provider: str | None = None
    judge_model: str | None = None

    # Applied to reasoning models (gpt-oss, o-series). "low" keeps the visible
    # answer inside the token budget instead of losing it to chain-of-thought.
    reasoning_effort: str = "low"

    # --- Local models ------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_device: str = "auto"

    # --- Vector store ------------------------------------------------------
    vector_backend: str = "faiss"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "healthcare_chunks"

    # --- Retrieval ---------------------------------------------------------
    chunk_tokens: int = 500
    chunk_overlap_tokens: int = 75
    retrieval_top_k: int = 20
    rerank_top_n: int = 5
    hybrid_alpha: float = 0.5

    # --- Storage -----------------------------------------------------------
    s3_bucket: str | None = None
    aws_region: str = "us-east-1"
    api_base_url: str = "http://localhost:8000"

    # --- Paths -------------------------------------------------------------
    data_dir: Path = Field(default=DATA_DIR)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def metadata_dir(self) -> Path:
        return self.data_dir / "metadata"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "evaluation"

    def ensure_dirs(self) -> None:
        for d in (
            self.raw_dir,
            self.processed_dir,
            self.metadata_dir,
            self.index_dir,
            self.eval_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def resolve_device(self) -> str:
        """Pick the best available torch device for local models."""
        if self.embedding_device != "auto":
            return self.embedding_device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
