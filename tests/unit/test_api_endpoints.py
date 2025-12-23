"""Unit tests for API endpoints.

These tests focus on testing the endpoint logic directly without
requiring a full FastAPI application to be running.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from rag_service.api.v1.endpoints import health, ingest, query
from rag_service.api.v1.schemas import (
    DirectoryIngestRequest,
    FileIngestRequest,
    QueryRequest,
)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test the health endpoint returns healthy status."""
        result = await health.health_check()
        assert result == {"status": "healthy"}

    @pytest.mark.asyncio
    async def test_readiness_check_ready(self):
        """Test readiness check when service is ready."""
        with patch("rag_service.api.v1.endpoints.health.get_embedding_service") as mock_get:
            mock_service = MagicMock()
            mock_get.return_value = mock_service

            result = await health.readiness_check()
            assert result == {"status": "ready"}

    @pytest.mark.asyncio
    async def test_readiness_check_not_ready(self):
        """Test readiness check when service is not ready."""
        with patch("rag_service.api.v1.endpoints.health.get_embedding_service") as mock_get:
            mock_get.side_effect = Exception("Model not loaded")

            result = await health.readiness_check()
            assert result["status"] == "not_ready"
            assert "error" in result

    @pytest.mark.asyncio
    async def test_system_info(self):
        """Test system info endpoint returns configuration."""
        with (
            patch("rag_service.api.v1.endpoints.health.get_settings") as mock_settings,
            patch("rag_service.api.v1.endpoints.health.get_embedding_service") as mock_embed,
            patch("rag_service.api.v1.endpoints.health.get_vector_store") as mock_store,
        ):
            # Mock settings
            mock_settings.return_value = MagicMock(
                embedding_model="test-model",
                get_resolved_device=lambda: "cpu",
                vector_store_backend="faiss",
                chunk_size=512,
                chunk_overlap=50,
            )

            # Mock embedding service
            mock_embed_service = MagicMock()
            mock_embed_service.model_name = "test-model"
            mock_embed_service.embedding_dim = 384
            mock_embed_service.get_device_info.return_value = {"device": "cpu"}
            mock_embed.return_value = mock_embed_service

            # Mock vector store
            mock_store_instance = MagicMock()
            mock_store_instance.list_collections.return_value = ["collection1", "collection2"]
            mock_store.return_value = mock_store_instance

            result = await health.system_info()

            assert "version" in result
            assert "config" in result
            assert "embedding" in result
            assert "vector_store" in result
            assert result["vector_store"]["collection_count"] == 2


class TestIngestEndpoints:
    """Tests for ingest endpoints."""

    @pytest.fixture
    def temp_file(self):
        """Create a temporary test file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test document content for ingestion.")
            temp_path = Path(f.name)
        yield temp_path
        temp_path.unlink(missing_ok=True)

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "file1.txt").write_text("Content 1")
            (tmp_path / "file2.txt").write_text("Content 2")
            yield tmp_path

    @pytest.mark.asyncio
    async def test_ingest_file_success(self, temp_file):
        """Test successful file ingestion."""
        request = FileIngestRequest(path=str(temp_file), collection="test")

        with (
            patch("rag_service.api.v1.endpoints.ingest.get_chunker") as mock_chunker,
            patch("rag_service.api.v1.endpoints.ingest.get_embedding_service") as mock_embed,
            patch("rag_service.api.v1.endpoints.ingest.get_vector_store") as mock_store,
            patch("rag_service.api.v1.endpoints.ingest.get_graph_store"),
            patch("rag_service.api.v1.endpoints.ingest.get_entity_extractor"),
        ):
            # Mock chunker - return a list with at least one document
            from rag_service.core.retriever import Document

            mock_chunk = Document(
                text="Test document content for ingestion.",
                metadata={"source": str(temp_file)},
            )
            mock_chunker_instance = MagicMock()
            mock_chunker_instance.process_file.return_value = [mock_chunk]
            mock_chunker.return_value = mock_chunker_instance

            # Mock embedding service
            mock_embed_service = MagicMock()
            mock_embed_service.embed_documents.return_value = [
                [0.1] * 384
            ]  # Mock embedding for documents
            mock_embed.return_value = mock_embed_service

            # Mock vector store
            mock_store_instance = MagicMock()
            mock_store_instance.add_documents.return_value = ["doc1"]
            mock_store.return_value = mock_store_instance

            # Mock graph store (disabled)
            with patch("rag_service.api.v1.endpoints.ingest.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock(enable_graph_rag=False)
                result = await ingest.ingest_file(request)

                assert result.status == "success"
                assert result.files_processed == 1
                assert result.documents_processed >= 1

    @pytest.mark.asyncio
    async def test_ingest_file_not_found(self):
        """Test file ingestion with nonexistent file."""
        request = FileIngestRequest(path="/nonexistent/file.txt", collection="test")

        # Must mock all dependencies that are called before the path check
        with (
            patch("rag_service.api.v1.endpoints.ingest.get_settings") as mock_settings,
            patch("rag_service.api.v1.endpoints.ingest.get_chunker") as mock_chunker,
            patch("rag_service.api.v1.endpoints.ingest.get_embedding_service") as mock_embed,
            patch("rag_service.api.v1.endpoints.ingest.get_vector_store") as mock_store,
            patch("rag_service.api.v1.endpoints.ingest.Path") as mock_path,
        ):
            mock_settings.return_value = MagicMock(enable_graph_rag=False)
            mock_chunker.return_value = MagicMock()
            mock_embed.return_value = MagicMock()
            mock_store.return_value = MagicMock()

            # Mock Path.exists() to return False
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = False
            mock_path.return_value = mock_path_instance

            with pytest.raises(HTTPException) as exc_info:
                await ingest.ingest_file(request)

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_ingest_directory_success(self, temp_dir):
        """Test successful directory ingestion."""
        request = DirectoryIngestRequest(
            path=str(temp_dir),
            collection="test",
            recursive=True,
        )

        with (
            patch("rag_service.api.v1.endpoints.ingest.get_chunker") as mock_chunker,
            patch("rag_service.api.v1.endpoints.ingest.get_embedding_service") as mock_embed,
            patch("rag_service.api.v1.endpoints.ingest.get_vector_store") as mock_store,
            patch("rag_service.api.v1.endpoints.ingest.get_graph_store"),
            patch("rag_service.api.v1.endpoints.ingest.get_entity_extractor"),
            patch("rag_service.api.v1.endpoints.ingest.get_settings") as mock_settings,
        ):
            # Mock chunker - return documents for each file
            from rag_service.core.retriever import Document

            mock_chunk1 = Document(
                text="Content 1", metadata={"source": str(temp_dir / "file1.txt")}
            )
            mock_chunk2 = Document(
                text="Content 2", metadata={"source": str(temp_dir / "file2.txt")}
            )
            mock_chunker_instance = MagicMock()
            # process_directory should return multiple documents
            mock_chunker_instance.process_directory.return_value = [mock_chunk1, mock_chunk2]
            mock_chunker.return_value = mock_chunker_instance

            # Mock embedding service
            mock_embed_service = MagicMock()
            mock_embed_service.embed.return_value = [[0.1] * 384]
            mock_embed.return_value = mock_embed_service

            # Mock vector store
            mock_store_instance = MagicMock()
            mock_store_instance.add_documents.return_value = ["doc1", "doc2"]
            mock_store.return_value = mock_store_instance

            # Mock settings
            mock_settings.return_value = MagicMock(enable_graph_rag=False)

            result = await ingest.ingest_directory(request)

            assert result.status == "success"
            assert result.files_processed >= 2

    @pytest.mark.asyncio
    async def test_ingest_directory_not_found(self):
        """Test directory ingestion with nonexistent directory."""
        request = DirectoryIngestRequest(
            path="/nonexistent/dir",
            collection="test",
            recursive=False,
        )

        # Must mock all dependencies that are called before the path check
        with (
            patch("rag_service.api.v1.endpoints.ingest.get_settings") as mock_settings,
            patch("rag_service.api.v1.endpoints.ingest.get_chunker") as mock_chunker,
            patch("rag_service.api.v1.endpoints.ingest.get_embedding_service") as mock_embed,
            patch("rag_service.api.v1.endpoints.ingest.get_vector_store") as mock_store,
            patch("rag_service.api.v1.endpoints.ingest.Path") as mock_path,
        ):
            mock_settings.return_value = MagicMock(enable_graph_rag=False)
            mock_chunker.return_value = MagicMock()
            mock_embed.return_value = MagicMock()
            mock_store.return_value = MagicMock()

            # Mock Path.exists() to return False
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = False
            mock_path.return_value = mock_path_instance

            with pytest.raises(HTTPException) as exc_info:
                await ingest.ingest_directory(request)

            assert exc_info.value.status_code == 404


class TestQueryEndpoints:
    """Tests for query endpoints."""

    @pytest.mark.asyncio
    async def test_query_success(self):
        """Test successful query."""
        request = QueryRequest(
            question="What is MAVLink?",
            top_k=5,
            collection="test",
        )

        with (
            patch("rag_service.api.v1.endpoints.query.get_vector_store") as mock_store,
            patch("rag_service.api.v1.endpoints.query.get_embedding_service") as mock_embed,
            patch("rag_service.api.v1.endpoints.query.get_query_router") as mock_router,
        ):
            # Mock embedding service
            mock_embed_service = MagicMock()
            mock_embed_service.embed_query.return_value = [0.1] * 384  # Query embedding
            mock_embed.return_value = mock_embed_service

            # Mock vector store
            from rag_service.core.retriever import SearchResult

            mock_result = SearchResult(
                text="MAVLink is a protocol",
                metadata={"source": "test.pdf"},
                score=0.9,
                document_id="doc1",
            )
            mock_store_instance = MagicMock()
            mock_store_instance.search.return_value = [mock_result]
            mock_store_instance.list_collections.return_value = ["test"]
            mock_store.return_value = mock_store_instance

            # Mock router
            mock_router_instance = MagicMock()
            mock_router_instance.classify.return_value = "vector"
            mock_router.return_value = mock_router_instance

            result = await query.query_documents(request)

            assert len(result.chunks) > 0
            assert result.chunks[0].text == "MAVLink is a protocol"
            assert result.collection == "test"

    @pytest.mark.asyncio
    async def test_query_collection_not_found(self):
        """Test query with nonexistent collection."""
        request = QueryRequest(
            question="What is MAVLink?",
            top_k=5,
            collection="nonexistent",
        )

        # Mock all dependencies called before the collection check
        # _perform_vector_search calls get_embedding_service() then get_vector_store()
        with (
            patch("rag_service.api.v1.endpoints.query.get_embedding_service") as mock_embed,
            patch("rag_service.api.v1.endpoints.query.get_vector_store") as mock_store,
        ):
            # Mock embedding service
            mock_embed.return_value = MagicMock()

            # Mock vector store with collection that doesn't include 'nonexistent'
            mock_store_instance = MagicMock()
            mock_store_instance.list_collections.return_value = ["test"]
            mock_store.return_value = mock_store_instance

            with pytest.raises(HTTPException) as exc_info:
                await query.query_documents(request)

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_query_validation_empty_question(self):
        """Test query validation with empty question."""
        # This should be caught by Pydantic validation
        with pytest.raises(ValueError):  # Pydantic validation error
            QueryRequest(question="", top_k=5)

    @pytest.mark.asyncio
    async def test_query_hybrid_strategy(self):
        """Test query with hybrid strategy."""
        request = QueryRequest(
            question="Explain the relationship between ARM and TAKEOFF",
            top_k=5,
            collection="test",
            strategy="hybrid",
        )

        with (
            patch("rag_service.api.v1.endpoints.query.get_vector_store") as mock_store,
            patch("rag_service.api.v1.endpoints.query.get_embedding_service") as mock_embed,
            patch("rag_service.api.v1.endpoints.query.get_query_router") as mock_router,
            patch("rag_service.api.v1.endpoints.query.get_graph_store") as mock_graph,
        ):
            # Mock embedding service
            mock_embed_service = MagicMock()
            mock_embed_service.embed_query.return_value = [0.1] * 384  # Query embedding
            mock_embed.return_value = mock_embed_service

            # Mock vector store
            from rag_service.core.retriever import SearchResult

            mock_result = SearchResult(
                text="ARM enables TAKEOFF",
                metadata={"source": "test.pdf"},
                score=0.9,
                document_id="doc1",
            )
            mock_store_instance = MagicMock()
            mock_store_instance.search.return_value = [mock_result]
            mock_store_instance.list_collections.return_value = ["test"]
            mock_store.return_value = mock_store_instance

            # Mock router
            mock_router_instance = MagicMock()
            mock_router_instance.classify.return_value = "hybrid"
            mock_router.return_value = mock_router_instance

            # Mock graph store
            mock_graph_instance = MagicMock()
            mock_graph_instance.query_entities.return_value = []
            mock_graph.return_value = mock_graph_instance

            # Note: query_documents returns QueryResponse, not HybridQueryResponse
            # For hybrid strategy, use query_hybrid endpoint instead
            result = await query.query_documents(request)

            assert len(result.chunks) > 0
            assert result.collection == "test"
