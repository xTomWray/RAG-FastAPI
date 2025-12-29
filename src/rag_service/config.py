"""Configuration management with YAML file support.

Priority (highest to lowest):
1. Environment variables
2. .env file
3. config.yaml file
4. Default values
"""

import logging
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Default config file locations (checked in order)
CONFIG_FILE_LOCATIONS = [
    Path("config.yaml"),
    Path("config.yml"),
    Path("./config/config.yaml"),
]


def find_config_file() -> Path | None:
    """Find the first existing config file.

    Returns:
        Path to config file if found, None otherwise.
    """
    for path in CONFIG_FILE_LOCATIONS:
        if path.exists():
            return path
    return None


def load_yaml_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Optional explicit path to config file.

    Returns:
        Dictionary of configuration values.
    """
    if config_path is None:
        config_path = find_config_file()

    if config_path is None or not config_path.exists():
        return {}

    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        return {}


def save_yaml_config(config: dict[str, Any], config_path: Path | None = None) -> Path:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary to save.
        config_path: Optional explicit path. Defaults to config.yaml in cwd.

    Returns:
        Path where config was saved.
    """
    if config_path is None:
        # Use existing config file or default to config.yaml
        config_path = find_config_file() or Path("config.yaml")

    # Ensure parent directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Add header comment
    header = """# RAG Documentation Service Configuration
# ==========================================
# This file contains user-editable settings for the RAG service.
# Settings here can be overridden by environment variables or .env file.
#
# Priority (highest to lowest):
#   1. Environment variables (e.g., EMBEDDING_MODEL=...)
#   2. .env file
#   3. This config.yaml file
#   4. Default values
#
# To use environment variables, convert setting names to UPPER_SNAKE_CASE:
#   embedding_model -> EMBEDDING_MODEL
#   chunk_size -> CHUNK_SIZE
#

"""

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info(f"Saved configuration to {config_path}")
    return config_path


class Settings(BaseSettings):
    """Application settings with YAML and environment variable support."""

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
        le=2048,
        description="Batch size for embedding generation",
    )

    # HuggingFace Hub Configuration
    hf_token: str | None = Field(
        default=None,
        description="HuggingFace API token for higher rate limits (optional)",
    )
    hf_model_cache_ttl: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Cache TTL in seconds for model search results",
    )

    # GPU Safeguard Configuration
    enable_gpu_safeguards: bool = Field(
        default=True,
        description="Enable GPU memory and temperature monitoring during embedding",
    )
    gpu_max_memory_percent: float = Field(
        default=80.0,
        ge=50.0,
        le=95.0,
        description="Maximum GPU memory usage (%) before throttling",
    )
    gpu_max_temperature_c: float = Field(
        default=75.0,
        ge=50.0,
        le=90.0,
        description="Maximum GPU temperature (°C) before throttling",
    )
    gpu_inter_batch_delay: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Seconds to pause between batches for cooling",
    )
    gpu_adaptive_batch_size: bool = Field(
        default=True,
        description="Automatically reduce batch size under memory/thermal pressure",
    )
    gpu_min_batch_size: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Minimum batch size when adapting",
    )
    gpu_power_limit_watts: int | None = Field(
        default=None,
        ge=100,
        le=600,
        description="GPU power limit in watts (requires admin). None to skip. Recommended: 280-320W for RTX 3090",
    )
    enable_gpu_warmup: bool = Field(
        default=True,
        description="Warm up GPU after model load to reduce power spikes",
    )
    precision: Literal["fp32", "fp16", "auto"] = Field(
        default="auto",
        description="Floating point precision. FP16 uses ~50% less memory and ~30% less power. Auto detects GPU capability.",
    )
    enable_tf32: bool = Field(
        default=True,
        description="Enable TF32 precision for matmul (Ampere+ GPUs). ~3x faster with <0.1% precision loss.",
    )
    enable_cudnn_benchmark: bool = Field(
        default=True,
        description="Enable cuDNN benchmark mode. Finds fastest algorithms for your GPU.",
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
    default_top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Default number of results (top_k) to return for queries",
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

    # Telemetry / Observability Configuration
    enable_telemetry: bool = Field(
        default=False,
        description="Enable OpenTelemetry tracing and metrics",
    )
    telemetry_service_name: str = Field(
        default="rag-documentation-service",
        description="Service name for telemetry identification",
    )
    telemetry_exporter: Literal["console", "otlp", "jaeger", "none"] = Field(
        default="console",
        description="Telemetry export backend. console=stdout, otlp=OpenTelemetry Protocol, jaeger=Jaeger",
    )
    telemetry_endpoint: str = Field(
        default="http://localhost:4317",
        description="OTLP/Jaeger collector endpoint",
    )
    telemetry_sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Trace sampling rate (1.0=all, 0.1=10% of requests)",
    )
    telemetry_log_spans: bool = Field(
        default=False,
        description="Log span start/end events (verbose, for debugging)",
    )
    telemetry_include_gpu_metrics: bool = Field(
        default=True,
        description="Include GPU memory/temperature in embedding spans",
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

    def to_yaml_dict(self) -> dict[str, Any]:
        """Convert settings to a dictionary suitable for YAML serialization.

        Excludes sensitive fields and converts Path objects to strings.
        """
        # Fields to exclude from config file (secrets should go in .env)
        sensitive_fields = {"neo4j_password"}

        result = {}
        for field_name in self.model_fields:
            if field_name in sensitive_fields:
                continue
            value = getattr(self, field_name)
            # Convert Path to string for YAML
            if isinstance(value, Path):
                value = str(value)
            result[field_name] = value
        return result


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


def _create_settings_with_yaml() -> Settings:
    """Create Settings instance with YAML config as base.

    Priority (highest to lowest):
    1. Environment variables
    2. .env file
    3. config.yaml file
    4. Default values
    """
    import os

    yaml_config = load_yaml_config() or {}

    # Get environment variables that override YAML
    # pydantic-settings uses UPPER_SNAKE_CASE for env vars
    env_overrides = {}
    for field_name in Settings.model_fields:
        env_name = field_name.upper()
        if env_name in os.environ:
            env_overrides[field_name] = os.environ[env_name]

    # Merge: YAML values first, then env overrides take precedence
    merged_config = {**yaml_config, **env_overrides}

    # Filter out empty strings from env (treat as "not set")
    merged_config = {k: v for k, v in merged_config.items() if v != ""}

    if merged_config:
        return Settings(**merged_config)

    return Settings()


# Cache for settings - can be cleared to reload
_settings_cache: Settings | None = None


def get_settings() -> Settings:
    """Get application settings (cached).

    Loads from YAML config file, then applies environment variable overrides.

    Returns:
        Settings: Application settings instance.
    """
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = _create_settings_with_yaml()
    return _settings_cache


def reload_settings() -> Settings:
    """Force reload settings from config files.

    Clears the cache and reloads from YAML and environment.

    Returns:
        Settings: Fresh settings instance.
    """
    global _settings_cache
    _settings_cache = None
    return get_settings()


def get_config_file_path() -> Path | None:
    """Get the path to the current config file.

    Returns:
        Path to config file if it exists, None otherwise.
    """
    return find_config_file()
