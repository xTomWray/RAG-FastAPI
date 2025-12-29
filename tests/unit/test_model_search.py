"""Unit tests for HuggingFace model search endpoints."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from rag_service.api.v1.endpoints import models
from rag_service.api.v1.schemas.models import ModelInfo, ModelSearchRequest


class TestModelSearchEndpoint:
    """Tests for the model search endpoint."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear the model cache before each test."""
        models._model_search_cache.clear()
        yield
        models._model_search_cache.clear()

    @pytest.fixture
    def mock_hf_response(self):
        """Sample HuggingFace API response."""
        return [
            {
                "id": "sentence-transformers/all-MiniLM-L6-v2",
                "downloads": 1000000,
                "likes": 500,
                "tags": ["sentence-transformers", "pytorch", "sentence-similarity"],
                "pipeline_tag": "sentence-similarity",
            },
            {
                "id": "sentence-transformers/paraphrase-MiniLM-L6-v2",
                "downloads": 500000,
                "likes": 200,
                "tags": ["sentence-transformers", "pytorch"],
                "pipeline_tag": "sentence-similarity",
            },
        ]

    @pytest.mark.asyncio
    async def test_search_models_success(self, mock_hf_response):
        """Test successful model search."""
        request = ModelSearchRequest(
            query="MiniLM",
            limit=10,
            filter_sentence_transformers=True,
        )

        with (
            patch("rag_service.api.v1.endpoints.models.httpx.AsyncClient") as mock_client,
            patch("rag_service.api.v1.endpoints.models.get_settings") as mock_settings,
        ):
            # Mock settings
            mock_settings.return_value = MagicMock(hf_token=None)

            # Mock httpx response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_hf_response
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await models.search_models(request)

            assert result.total == 2
            assert result.query == "MiniLM"
            assert result.cached is False
            assert len(result.models) == 2
            assert result.models[0].id == "sentence-transformers/all-MiniLM-L6-v2"
            assert result.models[0].downloads == 1000000

    @pytest.mark.asyncio
    async def test_search_models_cached(self):
        """Test that cached results are returned."""
        request = ModelSearchRequest(
            query="cached-test",
            limit=10,
        )

        # Pre-populate cache
        cache_key = models._make_cache_key(
            request.query,
            request.limit,
            request.filter_sentence_transformers,
            request.sort,
        )
        cached_models = [ModelInfo(id="cached/model", downloads=100, likes=10, tags=[])]
        models._set_cache(cache_key, cached_models)

        result = await models.search_models(request)

        assert result.cached is True
        assert len(result.models) == 1
        assert result.models[0].id == "cached/model"

    @pytest.mark.asyncio
    async def test_search_models_timeout(self):
        """Test handling of API timeout."""
        request = ModelSearchRequest(query="timeout-test")

        with (
            patch("rag_service.api.v1.endpoints.models.httpx.AsyncClient") as mock_client,
            patch("rag_service.api.v1.endpoints.models.get_settings") as mock_settings,
        ):
            mock_settings.return_value = MagicMock(hf_token=None)

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(
                side_effect=httpx.TimeoutException("Connection timeout")
            )
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(HTTPException) as exc_info:
                await models.search_models(request)

            assert exc_info.value.status_code == 504
            assert "timeout" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_search_models_api_error(self):
        """Test handling of HuggingFace API error."""
        request = ModelSearchRequest(query="error-test")

        with (
            patch("rag_service.api.v1.endpoints.models.httpx.AsyncClient") as mock_client,
            patch("rag_service.api.v1.endpoints.models.get_settings") as mock_settings,
        ):
            mock_settings.return_value = MagicMock(hf_token=None)

            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.text = "Rate limit exceeded"
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Rate limit",
                request=MagicMock(),
                response=mock_response,
            )

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(HTTPException) as exc_info:
                await models.search_models(request)

            assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_search_models_with_token(self, mock_hf_response):
        """Test that HF token is included in request headers."""
        request = ModelSearchRequest(query="test")

        with (
            patch("rag_service.api.v1.endpoints.models.httpx.AsyncClient") as mock_client,
            patch("rag_service.api.v1.endpoints.models.get_settings") as mock_settings,
        ):
            mock_settings.return_value = MagicMock(hf_token="test-token-123")

            mock_response = MagicMock()
            mock_response.json.return_value = mock_hf_response
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            await models.search_models(request)

            # Verify the get call included authorization header
            call_kwargs = mock_client_instance.get.call_args
            headers = call_kwargs.kwargs.get("headers", {})
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer test-token-123"

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Test cache clearing."""
        # Add entries to cache
        models._model_search_cache["test1"] = (time.time(), [])
        models._model_search_cache["test2"] = (time.time(), [])

        result = await models.clear_model_cache()

        assert "Cleared 2" in result["message"]
        assert len(models._model_search_cache) == 0

    @pytest.mark.asyncio
    async def test_get_cache_stats(self):
        """Test getting cache statistics."""
        # Add entry to cache
        models._model_search_cache["test"] = (
            time.time(),
            [ModelInfo(id="test/model", downloads=0, likes=0, tags=[])],
        )

        result = await models.get_cache_stats()

        assert result["cache_size"] == 1
        assert result["max_size"] == models.MAX_CACHE_SIZE
        assert result["ttl_seconds"] == models.CACHE_TTL_SECONDS
        assert len(result["entries"]) == 1
        assert result["entries"][0]["model_count"] == 1


class TestCacheHelpers:
    """Tests for cache helper functions."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear the model cache before each test."""
        models._model_search_cache.clear()
        yield
        models._model_search_cache.clear()

    def test_make_cache_key_consistent(self):
        """Test that cache keys are consistent for same inputs."""
        key1 = models._make_cache_key("test", 10, True, "downloads")
        key2 = models._make_cache_key("test", 10, True, "downloads")
        key3 = models._make_cache_key("TEST", 10, True, "downloads")  # Case insensitive

        assert key1 == key2
        assert key1 == key3  # Should normalize to lowercase

    def test_make_cache_key_differs(self):
        """Test that different inputs produce different keys."""
        key1 = models._make_cache_key("test", 10, True, "downloads")
        key2 = models._make_cache_key("test", 20, True, "downloads")
        key3 = models._make_cache_key("test", 10, False, "downloads")
        key4 = models._make_cache_key("test", 10, True, "likes")

        assert key1 != key2
        assert key1 != key3
        assert key1 != key4

    def test_cache_expiration(self):
        """Test cache entry expiration."""
        # Set with expired timestamp
        models._model_search_cache["expired"] = (
            time.time() - models.CACHE_TTL_SECONDS - 1,
            [],
        )

        result = models._get_from_cache("expired")
        assert result is None
        assert "expired" not in models._model_search_cache

    def test_cache_not_expired(self):
        """Test that non-expired entries are returned."""
        test_models = [ModelInfo(id="test/model", downloads=0, likes=0, tags=[])]
        models._model_search_cache["valid"] = (time.time(), test_models)

        result = models._get_from_cache("valid")
        assert result is not None
        assert len(result) == 1
        assert result[0].id == "test/model"

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        # Fill cache to max
        for i in range(models.MAX_CACHE_SIZE):
            models._set_cache(f"entry_{i}", [])
            time.sleep(0.001)  # Ensure different timestamps

        assert len(models._model_search_cache) == models.MAX_CACHE_SIZE

        # Add one more - should evict oldest
        models._set_cache("new_entry", [])

        assert len(models._model_search_cache) == models.MAX_CACHE_SIZE
        assert "new_entry" in models._model_search_cache
        assert "entry_0" not in models._model_search_cache  # Oldest evicted
