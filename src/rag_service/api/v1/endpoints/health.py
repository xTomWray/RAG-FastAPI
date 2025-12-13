"""Health check endpoints."""

from typing import Any

from fastapi import APIRouter

from rag_service.config import get_settings
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

