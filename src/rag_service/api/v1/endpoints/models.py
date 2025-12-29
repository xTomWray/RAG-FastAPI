"""Model search endpoints for HuggingFace Hub integration."""

import hashlib
import logging
import time
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query

from rag_service.api.v1.schemas.models import (
    ModelInfo,
    ModelSearchRequest,
    ModelSearchResponse,
)
from rag_service.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["models"])

# Simple in-memory cache with TTL
# Structure: {cache_key: (timestamp, data)}
_model_search_cache: dict[str, tuple[float, list[ModelInfo]]] = {}
CACHE_TTL_SECONDS = 3600  # 1 hour
MAX_CACHE_SIZE = 100

# HuggingFace API base URL
HF_API_URL = "https://huggingface.co/api/models"


def _make_cache_key(query: str, limit: int, filter_st: bool, sort: str) -> str:
    """Generate a cache key from search parameters.

    Args:
        query: Search query (case-insensitive).
        limit: Maximum results.
        filter_st: Whether filtering for sentence-transformers.
        sort: Sort field.

    Returns:
        16-character hex hash as cache key.
    """
    content = f"{query.lower()}:{limit}:{filter_st}:{sort}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _get_from_cache(key: str) -> list[ModelInfo] | None:
    """Get cached result if not expired.

    Args:
        key: Cache key to look up.

    Returns:
        Cached model list or None if expired/missing.
    """
    if key in _model_search_cache:
        timestamp, data = _model_search_cache[key]
        if time.time() - timestamp < CACHE_TTL_SECONDS:
            return data
        # Expired - remove it
        del _model_search_cache[key]
    return None


def _set_cache(key: str, data: list[ModelInfo]) -> None:
    """Store result in cache with LRU eviction.

    Args:
        key: Cache key.
        data: Model list to cache.
    """
    # Simple LRU: remove oldest entries if over limit
    if len(_model_search_cache) >= MAX_CACHE_SIZE:
        oldest_key = min(
            _model_search_cache.keys(), key=lambda k: _model_search_cache[k][0]
        )
        del _model_search_cache[oldest_key]

    _model_search_cache[key] = (time.time(), data)


async def _search_huggingface_models(
    query: str,
    limit: int,
    filter_sentence_transformers: bool,
    sort: str,
) -> list[ModelInfo]:
    """Search HuggingFace Hub API for models.

    Args:
        query: Search query string.
        limit: Maximum results.
        filter_sentence_transformers: Whether to filter by library.
        sort: Sort field (downloads, likes, lastModified).

    Returns:
        List of ModelInfo objects.

    Raises:
        HTTPException: If HuggingFace API fails.
    """
    settings = get_settings()

    # Build request parameters
    params: dict[str, Any] = {
        "search": query,
        "limit": limit,
        "sort": sort,
        "direction": "-1",  # Descending
    }

    if filter_sentence_transformers:
        # Use pipeline_tag instead of library filter to catch all embedding models
        # including Qwen, BGE, etc. that may use transformers library but support embeddings
        params["pipeline_tag"] = "feature-extraction"

    # Build headers with optional auth token
    headers: dict[str, str] = {}
    hf_token = getattr(settings, "hf_token", None)
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                HF_API_URL,
                params=params,
                headers=headers,
                timeout=10.0,
            )
            response.raise_for_status()

            models_data = response.json()

            return [
                ModelInfo(
                    id=m.get("id", ""),
                    downloads=m.get("downloads", 0),
                    likes=m.get("likes", 0),
                    tags=m.get("tags", []),
                    pipeline_tag=m.get("pipeline_tag"),
                )
                for m in models_data
            ]

        except httpx.TimeoutException:
            logger.warning("HuggingFace API timeout for query: %s", query)
            raise HTTPException(status_code=504, detail="HuggingFace API timeout")
        except httpx.HTTPStatusError as e:
            logger.warning(
                "HuggingFace API error: %s - %s", e.response.status_code, e.response.text
            )
            raise HTTPException(
                status_code=502,
                detail=f"HuggingFace API error: {e.response.status_code}",
            )


@router.post("/models/search", response_model=ModelSearchResponse)
async def search_models(request: ModelSearchRequest) -> ModelSearchResponse:
    """Search HuggingFace Hub for embedding models.

    Searches the HuggingFace Hub API for models matching the query.
    Results are cached for 1 hour to reduce API calls.

    Args:
        request: Search request with query and options.

    Returns:
        ModelSearchResponse with matching models.
    """
    cache_key = _make_cache_key(
        request.query,
        request.limit,
        request.filter_sentence_transformers,
        request.sort,
    )

    # Check cache first
    cached_result = _get_from_cache(cache_key)
    if cached_result is not None:
        logger.debug("Cache hit for model search: %s", request.query)
        return ModelSearchResponse(
            models=cached_result,
            total=len(cached_result),
            query=request.query,
            cached=True,
        )

    # Fetch from HuggingFace API
    models = await _search_huggingface_models(
        query=request.query,
        limit=request.limit,
        filter_sentence_transformers=request.filter_sentence_transformers,
        sort=request.sort,
    )

    # Cache the result
    _set_cache(cache_key, models)

    logger.info(
        "Model search completed: query=%s, results=%d",
        request.query,
        len(models),
    )

    return ModelSearchResponse(
        models=models,
        total=len(models),
        query=request.query,
        cached=False,
    )


@router.get("/models/search", response_model=ModelSearchResponse)
async def search_models_get(
    query: str = Query(
        ..., min_length=1, max_length=200, description="Search query for model names"
    ),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum results to return"),
    filter_st: bool = Query(
        default=True,
        alias="filter_st",
        description="Filter to sentence-transformer compatible models",
    ),
    sort: str = Query(
        default="downloads", description="Sort field: downloads, likes, lastModified"
    ),
) -> ModelSearchResponse:
    """Search HuggingFace Hub for embedding models (GET endpoint).

    Same as POST endpoint but accepts query parameters for easier
    browser and UI integration.

    Args:
        query: Search query for model names.
        limit: Maximum results to return.
        filter_st: Filter to sentence-transformer compatible models.
        sort: Sort field.

    Returns:
        ModelSearchResponse with matching models.
    """
    request = ModelSearchRequest(
        query=query,
        limit=limit,
        filter_sentence_transformers=filter_st,
        sort=sort,
    )
    return await search_models(request)


@router.delete("/models/cache")
async def clear_model_cache() -> dict[str, str]:
    """Clear the model search cache.

    Removes all cached search results. Useful after HuggingFace Hub
    updates or to force fresh results.

    Returns:
        Confirmation message with count of cleared entries.
    """
    global _model_search_cache
    count = len(_model_search_cache)
    _model_search_cache.clear()
    logger.info("Cleared model search cache: %d entries", count)
    return {"status": "ok", "message": f"Cleared {count} cached model searches"}


@router.get("/models/cache/stats")
async def get_cache_stats() -> dict[str, Any]:
    """Get model search cache statistics.

    Returns:
        Cache statistics including size, TTL, and entry ages.
    """
    now = time.time()
    entries = []

    for key, (timestamp, data) in _model_search_cache.items():
        age_seconds = int(now - timestamp)
        entries.append(
            {
                "key": key,
                "age_seconds": age_seconds,
                "expires_in": max(0, CACHE_TTL_SECONDS - age_seconds),
                "model_count": len(data),
            }
        )

    return {
        "cache_size": len(_model_search_cache),
        "max_size": MAX_CACHE_SIZE,
        "ttl_seconds": CACHE_TTL_SECONDS,
        "entries": sorted(entries, key=lambda x: x["age_seconds"]),
    }
