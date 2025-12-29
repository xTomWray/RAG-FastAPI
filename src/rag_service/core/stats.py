"""Runtime statistics collection and reporting.

Provides comprehensive metrics collection for:
- Service health and uptime
- GPU/CPU resource utilization
- Embedding performance metrics
- Vector store operations
- Query latency and throughput

Usage:
    from rag_service.core.stats import get_stats_collector, ServiceStats

    # Get stats for display
    stats = get_stats_collector()
    summary = stats.get_summary()

    # Record an operation
    stats.record_embedding(
        num_texts=100,
        duration_ms=250.5,
        batch_size=32,
    )
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Maximum history size for rolling metrics
MAX_HISTORY_SIZE = 1000


@dataclass
class OperationStats:
    """Statistics for a single operation type."""

    count: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float("inf")
    max_duration_ms: float = 0.0
    last_duration_ms: float = 0.0
    failures: int = 0
    last_timestamp: float = 0.0

    # Rolling history for percentile calculations
    recent_durations: deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def record(self, duration_ms: float, success: bool = True) -> None:
        """Record an operation execution."""
        self.count += 1
        self.total_duration_ms += duration_ms
        self.last_duration_ms = duration_ms
        self.last_timestamp = time.time()
        self.recent_durations.append(duration_ms)

        if duration_ms < self.min_duration_ms:
            self.min_duration_ms = duration_ms
        if duration_ms > self.max_duration_ms:
            self.max_duration_ms = duration_ms

        if not success:
            self.failures += 1

    @property
    def avg_duration_ms(self) -> float:
        """Average duration across all operations."""
        if self.count == 0:
            return 0.0
        return self.total_duration_ms / self.count

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.count == 0:
            return 100.0
        return ((self.count - self.failures) / self.count) * 100

    def get_percentile(self, p: int) -> float:
        """Get p-th percentile from recent durations."""
        if not self.recent_durations:
            return 0.0
        sorted_durations = sorted(self.recent_durations)
        idx = int(len(sorted_durations) * p / 100)
        return float(sorted_durations[min(idx, len(sorted_durations) - 1)])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "count": self.count,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "avg_duration_ms": round(self.avg_duration_ms, 2),
            "min_duration_ms": round(self.min_duration_ms, 2)
            if self.min_duration_ms != float("inf")
            else 0.0,
            "max_duration_ms": round(self.max_duration_ms, 2),
            "last_duration_ms": round(self.last_duration_ms, 2),
            "p50_duration_ms": round(self.get_percentile(50), 2),
            "p95_duration_ms": round(self.get_percentile(95), 2),
            "p99_duration_ms": round(self.get_percentile(99), 2),
            "failures": self.failures,
            "success_rate": round(self.success_rate, 1),
            "ops_per_second": round(
                self.count / max(1, time.time() - self.last_timestamp + 0.001), 2
            )
            if self.count > 0
            else 0.0,
        }


@dataclass
class EmbeddingStats:
    """Extended statistics for embedding operations.

    Tracks chunk embedding performance - chunks are pieces of documents
    after splitting, which get embedded into vectors.
    """

    ops: OperationStats = field(default_factory=OperationStats)
    total_chunks: int = 0  # Number of chunks embedded
    total_chars: int = 0  # Total characters processed
    total_batches: int = 0
    avg_batch_size: float = 0.0

    def record(
        self,
        duration_ms: float,
        num_texts: int,  # Actually chunks
        batch_size: int,
        char_count: int = 0,
        success: bool = True,
    ) -> None:
        """Record an embedding operation."""
        self.ops.record(duration_ms, success)
        self.total_chunks += num_texts
        self.total_chars += char_count
        self.total_batches += 1

        # Update running average batch size
        self.avg_batch_size = (
            self.avg_batch_size * (self.total_batches - 1) + batch_size
        ) / self.total_batches

    @property
    def chunks_per_second(self) -> float:
        """Average chunks embedded per second."""
        if self.ops.total_duration_ms == 0:
            return 0.0
        return self.total_chunks / (self.ops.total_duration_ms / 1000)

    @property
    def avg_chunk_chars(self) -> float:
        """Average characters per chunk."""
        if self.total_chunks == 0:
            return 0.0
        return self.total_chars / self.total_chunks

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            **self.ops.to_dict(),
            "total_chunks": self.total_chunks,
            "total_chars": self.total_chars,
            "avg_chunk_chars": round(self.avg_chunk_chars, 0),
            "total_batches": self.total_batches,
            "avg_batch_size": round(self.avg_batch_size, 1),
            "chunks_per_second": round(self.chunks_per_second, 1),
            # Legacy keys for compatibility
            "total_texts": self.total_chunks,
            "texts_per_second": round(self.chunks_per_second, 1),
        }


@dataclass
class SearchStats:
    """Extended statistics for search operations."""

    ops: OperationStats = field(default_factory=OperationStats)
    total_queries: int = 0
    total_results: int = 0
    avg_results_per_query: float = 0.0
    strategy_counts: dict[str, int] = field(default_factory=dict)

    def record(
        self,
        duration_ms: float,
        results_count: int,
        strategy: str = "vector",
        success: bool = True,
    ) -> None:
        """Record a search operation."""
        self.ops.record(duration_ms, success)
        self.total_queries += 1
        self.total_results += results_count

        # Update strategy counts
        self.strategy_counts[strategy] = self.strategy_counts.get(strategy, 0) + 1

        # Update running average
        self.avg_results_per_query = self.total_results / self.total_queries

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            **self.ops.to_dict(),
            "total_queries": self.total_queries,
            "total_results": self.total_results,
            "avg_results_per_query": round(self.avg_results_per_query, 1),
            "strategy_counts": self.strategy_counts,
        }


@dataclass
class RerankStats:
    """Extended statistics for reranking operations."""

    ops: OperationStats = field(default_factory=OperationStats)
    total_documents_reranked: int = 0
    total_rerank_calls: int = 0

    def record(
        self,
        duration_ms: float,
        num_documents: int,
        success: bool = True,
    ) -> None:
        """Record a rerank operation."""
        self.ops.record(duration_ms, success)
        self.total_documents_reranked += num_documents
        self.total_rerank_calls += 1

    @property
    def avg_docs_per_rerank(self) -> float:
        """Average documents per rerank call."""
        if self.total_rerank_calls == 0:
            return 0.0
        return self.total_documents_reranked / self.total_rerank_calls

    @property
    def docs_per_second(self) -> float:
        """Average documents reranked per second."""
        if self.ops.total_duration_ms == 0:
            return 0.0
        return self.total_documents_reranked / (self.ops.total_duration_ms / 1000)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            **self.ops.to_dict(),
            "total_documents_reranked": self.total_documents_reranked,
            "total_rerank_calls": self.total_rerank_calls,
            "avg_docs_per_rerank": round(self.avg_docs_per_rerank, 1),
            "docs_per_second": round(self.docs_per_second, 1),
        }


@dataclass
class IngestStats:
    """Extended statistics for document ingestion."""

    ops: OperationStats = field(default_factory=OperationStats)
    total_documents: int = 0
    total_chunks: int = 0
    total_bytes: int = 0
    file_type_counts: dict[str, int] = field(default_factory=dict)

    def record(
        self,
        duration_ms: float,
        documents: int,
        chunks: int,
        bytes_processed: int = 0,
        file_type: str = "unknown",
        success: bool = True,
    ) -> None:
        """Record an ingestion operation."""
        self.ops.record(duration_ms, success)
        self.total_documents += documents
        self.total_chunks += chunks
        self.total_bytes += bytes_processed
        self.file_type_counts[file_type] = self.file_type_counts.get(file_type, 0) + 1

    @property
    def avg_chunks_per_doc(self) -> float:
        """Average chunks per document."""
        if self.total_documents == 0:
            return 0.0
        return self.total_chunks / self.total_documents

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            **self.ops.to_dict(),
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "total_bytes": self.total_bytes,
            "total_mb": round(self.total_bytes / (1024 * 1024), 2),
            "avg_chunks_per_doc": round(self.avg_chunks_per_doc, 1),
            "file_type_counts": self.file_type_counts,
        }


class StatsCollector:
    """Central statistics collector for the RAG service.

    Thread-safe collector that aggregates metrics from all components.
    Uses RLock (reentrant lock) to allow nested lock acquisition within
    the same thread, preventing deadlocks when methods like get_summary()
    call other methods that also acquire the lock.
    """

    _instance: "StatsCollector | None" = None
    _class_lock = threading.Lock()  # For singleton creation only
    _initialized: bool = False

    def __new__(cls) -> "StatsCollector":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        # Use RLock to allow reentrant locking (same thread can acquire multiple times)
        # This prevents deadlock when get_summary() -> get_gpu_info() -> record_gpu_sample()
        self._lock = threading.RLock()
        self._start_time = time.time()

        # Core operation stats
        self.embeddings = EmbeddingStats()
        self.searches = SearchStats()
        self.reranks = RerankStats()
        self.ingestions = IngestStats()

        # General operation counters
        self.api_requests = OperationStats()
        self.chunking = OperationStats()
        self.vector_store = OperationStats()
        self.graph_store = OperationStats()

        # GPU tracking (updated periodically)
        self._gpu_samples: deque[dict[str, Any]] = deque(maxlen=60)  # Last 60 samples
        self._last_gpu_sample: dict[str, Any] = {}

        # Error tracking
        self._errors: deque[dict[str, Any]] = deque(maxlen=50)  # Last 50 errors

        self._initialized = True
        logger.debug("Stats collector initialized")

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._start_time = time.time()
            self.embeddings = EmbeddingStats()
            self.searches = SearchStats()
            self.reranks = RerankStats()
            self.ingestions = IngestStats()
            self.api_requests = OperationStats()
            self.chunking = OperationStats()
            self.vector_store = OperationStats()
            self.graph_store = OperationStats()
            self._gpu_samples.clear()
            self._errors.clear()
            logger.info("Stats collector reset")

    # === Recording Methods ===

    def record_embedding(
        self,
        duration_ms: float,
        num_texts: int,
        batch_size: int = 32,
        char_count: int = 0,
        success: bool = True,
    ) -> None:
        """Record an embedding operation."""
        with self._lock:
            self.embeddings.record(
                duration_ms=duration_ms,
                num_texts=num_texts,
                batch_size=batch_size,
                char_count=char_count,
                success=success,
            )

    def record_search(
        self,
        duration_ms: float,
        results_count: int,
        strategy: str = "vector",
        success: bool = True,
    ) -> None:
        """Record a search operation."""
        with self._lock:
            self.searches.record(
                duration_ms=duration_ms,
                results_count=results_count,
                strategy=strategy,
                success=success,
            )

    def record_rerank(
        self,
        duration_ms: float,
        num_documents: int,
        success: bool = True,
    ) -> None:
        """Record a reranking operation."""
        with self._lock:
            self.reranks.record(
                duration_ms=duration_ms,
                num_documents=num_documents,
                success=success,
            )

    def record_ingestion(
        self,
        duration_ms: float,
        documents: int = 1,
        chunks: int = 0,
        bytes_processed: int = 0,
        file_type: str = "unknown",
        success: bool = True,
    ) -> None:
        """Record a document ingestion."""
        with self._lock:
            self.ingestions.record(
                duration_ms=duration_ms,
                documents=documents,
                chunks=chunks,
                bytes_processed=bytes_processed,
                file_type=file_type,
                success=success,
            )

    def record_api_request(self, duration_ms: float, success: bool = True) -> None:
        """Record an API request."""
        with self._lock:
            self.api_requests.record(duration_ms, success)

    def record_chunking(self, duration_ms: float, success: bool = True) -> None:
        """Record a chunking operation."""
        with self._lock:
            self.chunking.record(duration_ms, success)

    def record_vector_store_op(self, duration_ms: float, success: bool = True) -> None:
        """Record a vector store operation."""
        with self._lock:
            self.vector_store.record(duration_ms, success)

    def record_graph_store_op(self, duration_ms: float, success: bool = True) -> None:
        """Record a graph store operation."""
        with self._lock:
            self.graph_store.record(duration_ms, success)

    def record_error(
        self, error_type: str, message: str, context: dict[str, Any] | None = None
    ) -> None:
        """Record an error."""
        with self._lock:
            self._errors.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "type": error_type,
                    "message": message[:500],  # Truncate long messages
                    "context": context or {},
                }
            )

    def record_gpu_sample(self, gpu_info: dict[str, Any]) -> None:
        """Record a GPU metrics sample."""
        with self._lock:
            sample = {
                "timestamp": time.time(),
                **gpu_info,
            }
            self._gpu_samples.append(sample)
            self._last_gpu_sample = sample

    # === Query Methods ===

    @property
    def uptime_seconds(self) -> float:
        """Service uptime in seconds."""
        return time.time() - self._start_time

    @property
    def uptime_formatted(self) -> str:
        """Service uptime in human-readable format."""
        seconds = int(self.uptime_seconds)
        days, remainder = divmod(seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, secs = divmod(remainder, 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")

        return " ".join(parts)

    def get_gpu_info(self) -> dict[str, Any]:
        """Get current GPU information."""
        try:
            import torch

            if not torch.cuda.is_available():
                return {"available": False}

            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            info = {
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "memory_allocated_gb": round(memory_allocated, 2),
                "memory_reserved_gb": round(memory_reserved, 2),
                "memory_total_gb": round(memory_total, 2),
                "memory_percent": round((memory_allocated / memory_total) * 100, 1),
            }

            # Try to get nvidia-smi metrics
            try:
                import subprocess

                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=temperature.gpu,utilization.gpu,power.draw,power.limit",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(", ")
                    if len(parts) >= 4:
                        info["temperature_c"] = float(parts[0])
                        info["utilization_percent"] = float(parts[1])
                        info["power_draw_watts"] = float(parts[2])
                        info["power_limit_watts"] = float(parts[3])
            except Exception:
                pass

            self.record_gpu_sample(info)
            return info

        except ImportError:
            return {"available": False, "error": "PyTorch not installed"}
        except Exception as e:
            return {"available": False, "error": str(e)}

    def get_system_info(self) -> dict[str, Any]:
        """Get current system information."""
        import platform
        import sys

        info: dict[str, Any] = {
            "python_version": sys.version.split()[0],
            "platform": platform.system(),
            "platform_release": platform.release(),
            "architecture": platform.machine(),
        }

        # CPU info
        try:
            import os

            info["cpu_count"] = os.cpu_count() or 0
        except Exception:
            pass

        # Memory info
        try:
            import psutil

            mem = psutil.virtual_memory()
            info["memory_total_gb"] = round(mem.total / (1024**3), 1)
            info["memory_available_gb"] = round(mem.available / (1024**3), 1)
            info["memory_percent"] = mem.percent
        except ImportError:
            pass

        return info

    def get_config_info(self) -> dict[str, Any]:
        """Get current configuration info."""
        try:
            from rag_service.config import get_settings

            settings = get_settings()
            return {
                "embedding_model": settings.embedding_model,
                "device": settings.get_resolved_device(),
                "vector_store_backend": settings.vector_store_backend,
                "batch_size": settings.embedding_batch_size,
                "chunk_size": settings.chunk_size,
                "enable_graph_rag": settings.enable_graph_rag,
                "enable_telemetry": settings.enable_telemetry,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive stats summary."""
        with self._lock:
            return {
                "service": {
                    "uptime": self.uptime_formatted,
                    "uptime_seconds": round(self.uptime_seconds, 1),
                    "start_time": datetime.fromtimestamp(self._start_time).isoformat(),
                },
                "config": self.get_config_info(),
                "gpu": self.get_gpu_info(),
                "system": self.get_system_info(),
                "operations": {
                    "embeddings": self.embeddings.to_dict(),
                    "searches": self.searches.to_dict(),
                    "reranks": self.reranks.to_dict(),
                    "ingestions": self.ingestions.to_dict(),
                    "api_requests": self.api_requests.to_dict(),
                    "chunking": self.chunking.to_dict(),
                    "vector_store": self.vector_store.to_dict(),
                    "graph_store": self.graph_store.to_dict(),
                },
                "recent_errors": list(self._errors),
                "health": self._calculate_health(),
            }

    def _calculate_health(self) -> dict[str, Any]:
        """Calculate overall health metrics."""
        # Calculate health score (0-100)
        scores = []

        # API success rate
        if self.api_requests.count > 0:
            scores.append(self.api_requests.success_rate)

        # Embedding success rate
        if self.embeddings.ops.count > 0:
            scores.append(self.embeddings.ops.success_rate)

        # Search success rate
        if self.searches.ops.count > 0:
            scores.append(self.searches.ops.success_rate)

        # No errors is healthy
        error_penalty = min(len(self._errors) * 2, 20)

        overall_score = (sum(scores) / len(scores) if scores else 100.0) - error_penalty
        overall_score = max(0, min(100, overall_score))

        status = (
            "healthy" if overall_score >= 90 else "degraded" if overall_score >= 70 else "unhealthy"
        )

        return {
            "status": status,
            "score": round(overall_score, 1),
            "total_operations": (
                self.embeddings.ops.count + self.searches.ops.count + self.ingestions.ops.count
            ),
            "total_errors": len(self._errors),
        }

    def get_cli_summary(self) -> str:
        """Get formatted summary for CLI output."""
        summary = self.get_summary()

        lines = []
        lines.append("=" * 60)
        lines.append("RAG Documentation Service - Statistics")
        lines.append("=" * 60)

        # Service info
        lines.append("\n📊 Service Status:")
        lines.append(f"   Uptime: {summary['service']['uptime']}")
        lines.append(
            f"   Health: {summary['health']['status'].upper()} ({summary['health']['score']}%)"
        )
        lines.append(f"   Total Operations: {summary['health']['total_operations']:,}")

        # Configuration
        config = summary.get("config", {})
        lines.append("\n⚙️ Configuration:")
        lines.append(f"   Model: {config.get('embedding_model', 'N/A')}")
        lines.append(f"   Device: {config.get('device', 'N/A')}")
        lines.append(f"   Vector Store: {config.get('vector_store_backend', 'N/A')}")

        # GPU info
        gpu = summary.get("gpu", {})
        if gpu.get("available"):
            lines.append(f"\n🎮 GPU ({gpu.get('device_name', 'Unknown')}):")
            lines.append(
                f"   Memory: {gpu.get('memory_allocated_gb', 0):.1f} / {gpu.get('memory_total_gb', 0):.1f} GB ({gpu.get('memory_percent', 0):.0f}%)"
            )
            if "temperature_c" in gpu:
                lines.append(f"   Temperature: {gpu['temperature_c']:.0f}°C")
            if "power_draw_watts" in gpu:
                lines.append(
                    f"   Power: {gpu['power_draw_watts']:.0f}W / {gpu.get('power_limit_watts', 'N/A')}W"
                )
        else:
            lines.append("\n🖥️ Running on CPU")

        # Embeddings
        emb = summary["operations"]["embeddings"]
        if emb["count"] > 0:
            lines.append("\n🧠 Embeddings:")
            lines.append(f"   Operations: {emb['count']:,}")
            lines.append(f"   Chunks Embedded: {emb['total_chunks']:,}")
            lines.append(f"   Throughput: {emb['chunks_per_second']:.1f} chunks/sec")
            lines.append(f"   Avg Chunk Size: {emb['avg_chunk_chars']:.0f} chars")
            lines.append(
                f"   Latency: avg={emb['avg_duration_ms']:.1f}ms, p95={emb['p95_duration_ms']:.1f}ms"
            )

        # Searches
        search = summary["operations"]["searches"]
        if search["count"] > 0:
            lines.append("\n🔍 Searches:")
            lines.append(f"   Queries: {search['total_queries']:,}")
            lines.append(f"   Avg Results: {search['avg_results_per_query']:.1f}")
            lines.append(
                f"   Latency: avg={search['avg_duration_ms']:.1f}ms, p95={search['p95_duration_ms']:.1f}ms"
            )
            if search["strategy_counts"]:
                strategies = ", ".join(f"{k}={v}" for k, v in search["strategy_counts"].items())
                lines.append(f"   Strategies: {strategies}")

        # Ingestions
        ingest = summary["operations"]["ingestions"]
        if ingest["count"] > 0:
            lines.append("\n📥 Ingestions:")
            lines.append(f"   Documents: {ingest['total_documents']:,}")
            lines.append(f"   Chunks Created: {ingest['total_chunks']:,}")
            lines.append(f"   Data Processed: {ingest['total_mb']:.1f} MB")
            lines.append(f"   Avg Chunks/Doc: {ingest['avg_chunks_per_doc']:.1f}")

        # Errors
        errors = summary["recent_errors"]
        if errors:
            lines.append(f"\n⚠️ Recent Errors ({len(errors)}):")
            for err in errors[-3:]:  # Show last 3
                lines.append(f"   [{err['timestamp']}] {err['type']}: {err['message'][:50]}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# Singleton instance
_stats_collector: StatsCollector | None = None


def get_stats_collector() -> StatsCollector:
    """Get the global stats collector instance."""
    global _stats_collector
    if _stats_collector is None:
        _stats_collector = StatsCollector()
    return _stats_collector


def reset_stats() -> None:
    """Reset all collected statistics."""
    get_stats_collector().reset()


# Context manager for timing operations
class timed_operation:
    """Context manager for timing and recording operations.

    Usage:
        with timed_operation("embedding", num_texts=100, batch_size=32) as t:
            result = embed(texts)
        # Automatically records to stats collector
    """

    def __init__(
        self,
        operation: str,
        **kwargs: Any,
    ) -> None:
        self.operation = operation
        self.kwargs = kwargs
        self.start_time = 0.0
        self.success = True

    def __enter__(self) -> "timed_operation":
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        self.success = exc_type is None

        stats = get_stats_collector()

        if self.operation == "embedding":
            stats.record_embedding(
                duration_ms=duration_ms,
                num_texts=self.kwargs.get("num_texts", 0),
                batch_size=self.kwargs.get("batch_size", 32),
                char_count=self.kwargs.get("char_count", 0),
                success=self.success,
            )
        elif self.operation == "search":
            stats.record_search(
                duration_ms=duration_ms,
                results_count=self.kwargs.get("results_count", 0),
                strategy=self.kwargs.get("strategy", "vector"),
                success=self.success,
            )
        elif self.operation == "ingestion":
            stats.record_ingestion(
                duration_ms=duration_ms,
                documents=self.kwargs.get("documents", 1),
                chunks=self.kwargs.get("chunks", 0),
                bytes_processed=self.kwargs.get("bytes_processed", 0),
                file_type=self.kwargs.get("file_type", "unknown"),
                success=self.success,
            )
        elif self.operation == "api_request":
            stats.record_api_request(duration_ms, self.success)
        elif self.operation == "chunking":
            stats.record_chunking(duration_ms, self.success)
        elif self.operation == "vector_store":
            stats.record_vector_store_op(duration_ms, self.success)
        elif self.operation == "graph_store":
            stats.record_graph_store_op(duration_ms, self.success)

        if not self.success and exc_val:
            stats.record_error(
                error_type=exc_type.__name__ if exc_type else "Unknown",
                message=str(exc_val),
                context={"operation": self.operation, **self.kwargs},
            )
        # Don't suppress exceptions (return None)
