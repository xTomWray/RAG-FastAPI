"""Caching infrastructure for query results and embeddings.

Provides abstract cache interface with implementations for:
- InMemoryCache: Simple dict-based cache for single-instance deployments
- RedisCache: Distributed cache for scaled deployments (requires redis extra)

Example usage:
    from rag_service.infrastructure.cache import create_cache, make_cache_key

    cache = create_cache("memory")  # or "redis"

    # Cache a query embedding
    key = make_cache_key("embedding", query_text)
    cached = await cache.get(key)
    if cached is None:
        embedding = compute_embedding(query_text)
        await cache.set(key, embedding, ttl=3600)
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Literal

from rag_service.core.logging import get_logger

logger = get_logger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    async def get(self, key: str) -> bytes | None:
        """Get a value from the cache.

        Args:
            key: Cache key.

        Returns:
            Cached value as bytes, or None if not found/expired.
        """
        ...

    @abstractmethod
    async def set(self, key: str, value: bytes, ttl: int = 3600) -> None:
        """Set a value in the cache.

        Args:
            key: Cache key.
            value: Value to cache (as bytes).
            ttl: Time-to-live in seconds.
        """
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a value from the cache.

        Args:
            key: Cache key to delete.
        """
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Clear all values from the cache."""
        ...


class InMemoryCache(CacheBackend):
    """Simple in-memory cache for single-instance deployments.

    Uses a dict with expiration timestamps. Not suitable for
    multi-process or distributed deployments.
    """

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize the in-memory cache.

        Args:
            max_size: Maximum number of entries before LRU eviction.
        """
        self._cache: dict[str, tuple[bytes, float]] = {}
        self._max_size = max_size

    async def get(self, key: str) -> bytes | None:
        """Get a value from the cache."""
        if key in self._cache:
            value, expires = self._cache[key]
            if time.time() < expires:
                return value
            # Expired - remove it
            del self._cache[key]
        return None

    async def set(self, key: str, value: bytes, ttl: int = 3600) -> None:
        """Set a value in the cache."""
        # Simple LRU eviction - remove oldest entries if over limit
        if len(self._cache) >= self._max_size:
            # Remove 10% of oldest entries
            entries = sorted(self._cache.items(), key=lambda x: x[1][1])
            for old_key, _ in entries[: self._max_size // 10]:
                del self._cache[old_key]

        self._cache[key] = (value, time.time() + ttl)

    async def delete(self, key: str) -> None:
        """Delete a value from the cache."""
        self._cache.pop(key, None)

    async def clear(self) -> None:
        """Clear all values from the cache."""
        self._cache.clear()


class RedisCache(CacheBackend):
    """Redis-based cache for distributed deployments.

    Requires the 'cache' optional extra:
        pip install rag-documentation-service[cache]
    """

    def __init__(self, url: str = "redis://localhost:6379") -> None:
        """Initialize the Redis cache.

        Args:
            url: Redis connection URL.
        """
        self._url = url
        self._redis: Any = None

    async def _get_redis(self) -> Any:
        """Lazily initialize Redis connection."""
        if self._redis is None:
            try:
                import redis.asyncio as redis

                self._redis = redis.from_url(self._url)
            except ImportError:
                raise ImportError(
                    "Redis support requires the 'cache' extra. "
                    "Install with: pip install rag-documentation-service[cache]"
                )
        return self._redis

    async def get(self, key: str) -> bytes | None:
        """Get a value from Redis."""
        redis = await self._get_redis()
        result = await redis.get(key)
        return result if result is None else bytes(result)

    async def set(self, key: str, value: bytes, ttl: int = 3600) -> None:
        """Set a value in Redis with expiration."""
        redis = await self._get_redis()
        await redis.setex(key, ttl, value)

    async def delete(self, key: str) -> None:
        """Delete a value from Redis."""
        redis = await self._get_redis()
        await redis.delete(key)

    async def clear(self) -> None:
        """Clear all values from Redis (flushdb)."""
        redis = await self._get_redis()
        await redis.flushdb()


class NoOpCache(CacheBackend):
    """No-operation cache that doesn't store anything.

    Useful for disabling caching without changing code.
    """

    async def get(self, _key: str) -> bytes | None:
        """Always returns None."""
        return None

    async def set(self, key: str, value: bytes, ttl: int = 3600) -> None:
        """Does nothing."""
        pass

    async def delete(self, key: str) -> None:
        """Does nothing."""
        pass

    async def clear(self) -> None:
        """Does nothing."""
        pass


def make_cache_key(prefix: str, *args: Any) -> str:
    """Generate a cache key from arguments.

    Creates a consistent hash-based key from the given prefix
    and arguments. Useful for caching query results.

    Args:
        prefix: Key prefix (e.g., "embedding", "query").
        *args: Arguments to include in the key.

    Returns:
        A cache key string like "embedding:a1b2c3d4".
    """
    content = json.dumps(args, sort_keys=True, default=str)
    hash_value = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"{prefix}:{hash_value}"


def create_cache(
    backend: Literal["memory", "redis", "none"] = "memory",
    redis_url: str = "redis://localhost:6379",
    max_size: int = 1000,
) -> CacheBackend:
    """Create a cache backend instance.

    Factory function for creating the appropriate cache backend
    based on configuration.

    Args:
        backend: Cache backend type.
        redis_url: Redis connection URL (for redis backend).
        max_size: Max entries for in-memory cache.

    Returns:
        A CacheBackend instance.
    """
    if backend == "redis":
        logger.info("Using Redis cache backend", url=redis_url)
        return RedisCache(url=redis_url)
    elif backend == "memory":
        logger.info("Using in-memory cache backend", max_size=max_size)
        return InMemoryCache(max_size=max_size)
    else:
        logger.info("Caching disabled")
        return NoOpCache()
