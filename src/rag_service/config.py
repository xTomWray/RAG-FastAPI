"""Configuration management using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Embedding Model Configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model ID for embeddings",
    )
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto",
        description="Device for model inference",
    )
    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Batch size for embedding generation",
    )

    # Vector Store Configuration
    vector_store_backend: Literal["faiss", "chroma"] = Field(
        default="faiss",
        description="Vector store backend to use",
    )
    faiss_index_dir: Path = Field(
        default=Path("./data/index"),
        description="Directory for FAISS index persistence",
    )
    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma"),
        description="Directory for ChromaDB persistence",
    )
    default_collection: str = Field(
        default="documents",
        description="Default collection name",
    )

    # Document Processing Configuration
    chunk_size: int = Field(
        default=512,
        ge=100,
        le=4096,
        description="Chunk size in characters",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlap between chunks",
    )
    pdf_strategy: Literal["fast", "hi_res", "ocr_only"] = Field(
        default="fast",
        description="PDF processing strategy",
    )

    # API Configuration
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8080, ge=1, le=65535, description="API port")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="CORS allowed origins",
    )
    api_prefix: str = Field(default="/api/v1", description="API route prefix")

    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )

    # Graph Store Configuration
    graph_store_backend: Literal["neo4j", "memory"] = Field(
        default="memory",
        description="Graph store backend to use",
    )
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j connection URI",
    )
    neo4j_user: str = Field(
        default="neo4j",
        description="Neo4j username",
    )
    neo4j_password: str = Field(
        default="password",
        description="Neo4j password",
    )
    neo4j_database: str = Field(
        default="neo4j",
        description="Neo4j database name",
    )

    # Query Router Configuration
    router_mode: Literal["pattern", "llm"] = Field(
        default="pattern",
        description="Query classification mode",
    )
    default_query_strategy: Literal["vector", "graph", "hybrid"] = Field(
        default="vector",
        description="Default strategy for ambiguous queries",
    )

    # Entity Extraction Configuration
    entity_extraction_mode: Literal["rule_based", "llm"] = Field(
        default="rule_based",
        description="Entity extraction mode",
    )
    entity_extraction_domain: Literal["general", "mavlink"] = Field(
        default="general",
        description="Domain specialization for extraction",
    )
    ollama_model: str = Field(
        default="llama3.2",
        description="Ollama model for LLM-based extraction/routing",
    )

    # GraphRAG Feature Flags
    enable_graph_rag: bool = Field(
        default=True,
        description="Enable GraphRAG features",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse CORS origins from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("faiss_index_dir", "chroma_persist_dir", mode="before")
    @classmethod
    def parse_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        return Path(v) if isinstance(v, str) else v

    def get_resolved_device(self) -> str:
        """Get the resolved device based on auto-detection or explicit setting."""
        if self.device != "auto":
            return self.device
        return detect_device()


def detect_device() -> str:
    """Auto-detect the best available compute device.

    Returns:
        str: "cuda" for NVIDIA GPUs, "mps" for Apple Silicon, "cpu" otherwise.
    """
    try:
        import torch

        # Check CUDA - must have actual GPU device, not just CUDA libraries
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            try:
                # Verify we can actually use the GPU
                torch.cuda.get_device_name(0)
                return "cuda"
            except Exception:
                pass
        # Check Apple Silicon MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings.

    Returns:
        Settings: Application settings instance.
    """
    return Settings()

