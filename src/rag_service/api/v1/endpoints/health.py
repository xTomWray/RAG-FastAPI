"""Health check and monitoring endpoints."""

from typing import Any

from fastapi import APIRouter, Query

from rag_service.config import get_settings
from rag_service.core.stats import get_stats_collector, reset_stats
from rag_service.dependencies import get_embedding_service, get_vector_store

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness probe endpoint.

    Returns:
        Simple health status.
    """
    return {"status": "healthy"}


@router.get("/ready")
async def readiness_check() -> dict[str, str]:
    """Readiness probe endpoint.

    Checks if the service is ready to serve requests.

    Returns:
        Readiness status.
    """
    try:
        # Try to get embedding service (validates model loading)
        _ = get_embedding_service()
        return {"status": "ready"}
    except Exception as e:
        return {"status": "not_ready", "error": str(e)}


@router.get("/info")
async def system_info() -> dict[str, Any]:
    """Get system information.

    Returns detailed information about the service configuration,
    loaded models, and available resources.

    Returns:
        System information dictionary.
    """
    settings = get_settings()

    info: dict[str, Any] = {
        "version": "0.1.0",
        "config": {
            "embedding_model": settings.embedding_model,
            "device": settings.get_resolved_device(),
            "vector_store_backend": settings.vector_store_backend,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
    }

    try:
        embedding_service = get_embedding_service()
        info["embedding"] = {
            "model": embedding_service.model_name,
            "dimension": embedding_service.embedding_dim,
            **embedding_service.get_device_info(),
        }
    except Exception as e:
        info["embedding"] = {"error": str(e)}

    try:
        vector_store = get_vector_store()
        collections = vector_store.list_collections()
        info["vector_store"] = {
            "backend": settings.vector_store_backend,
            "collections": collections,
            "collection_count": len(collections),
        }
    except Exception as e:
        info["vector_store"] = {"error": str(e)}

    return info


@router.get("/stats")
async def get_stats(
    include_gpu: bool = Query(default=True, description="Include GPU metrics"),
    include_errors: bool = Query(default=True, description="Include recent errors"),
) -> dict[str, Any]:
    """Get comprehensive service statistics.

    Returns detailed runtime statistics including:
    - Service uptime and health status
    - Operation counts and latencies (embeddings, searches, ingestions)
    - GPU/CPU resource utilization
    - Recent errors

    Args:
        include_gpu: Whether to include GPU metrics (may take ~100ms)
        include_errors: Whether to include recent error log

    Returns:
        Comprehensive statistics dictionary.
    """
    stats = get_stats_collector()
    summary = stats.get_summary()

    # Optionally exclude sections to reduce response size
    if not include_gpu:
        summary["gpu"] = {"available": summary.get("gpu", {}).get("available", False)}

    if not include_errors:
        summary["recent_errors"] = []

    return summary


@router.get("/stats/gpu")
async def get_gpu_stats() -> dict[str, Any]:
    """Get GPU-specific statistics.

    Returns current GPU metrics including memory, temperature,
    utilization, and power draw.

    Returns:
        GPU statistics dictionary.
    """
    stats = get_stats_collector()
    return stats.get_gpu_info()


@router.get("/stats/operations")
async def get_operation_stats() -> dict[str, Any]:
    """Get operation statistics only.

    Returns statistics for all operation types without
    GPU or system information.

    Returns:
        Operation statistics dictionary.
    """
    stats = get_stats_collector()
    summary = stats.get_summary()
    return {
        "uptime": summary["service"]["uptime"],
        "health": summary["health"],
        "operations": summary["operations"],
    }


@router.post("/stats/reset")
async def post_reset_stats() -> dict[str, str]:
    """Reset all statistics counters.

    Clears all accumulated statistics. Useful for benchmarking
    or after configuration changes.

    Returns:
        Confirmation message.
    """
    reset_stats()
    return {"status": "ok", "message": "Statistics reset successfully"}
