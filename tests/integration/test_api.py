"""Integration tests for the API endpoints."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_client(tmp_path: Path):
    """Create a test client with temporary storage directories."""
    import os

    # Set environment variables for test configuration
    os.environ["FAISS_INDEX_DIR"] = str(tmp_path / "faiss")
    os.environ["CHROMA_PERSIST_DIR"] = str(tmp_path / "chroma")
    os.environ["DEVICE"] = "cpu"
    os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"

    # Clear cached settings and services
    from rag_service.config import reload_settings
    from rag_service.dependencies import (
        get_chunker,
        get_embedding_service,
        get_graph_store,
        get_query_router,
        get_vector_store,
    )

    # Reload settings (clears manual cache)
    reload_settings()
    # Clear LRU caches
    get_embedding_service.cache_clear()
    get_vector_store.cache_clear()
    get_chunker.cache_clear()
    get_graph_store.cache_clear()
    get_query_router.cache_clear()

    from rag_service.main import create_app

    app = create_app()
    with TestClient(app) as client:
        yield client


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_endpoint(self, test_client: TestClient) -> None:
        """Test the health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_ready_endpoint(self, test_client: TestClient) -> None:
        """Test the readiness endpoint."""
        response = test_client.get("/ready")
        assert response.status_code == 200
        # May be "ready" or "not_ready" depending on model loading

    def test_info_endpoint(self, test_client: TestClient) -> None:
        """Test the system info endpoint."""
        response = test_client.get("/info")
        assert response.status_code == 200

        data = response.json()
        assert "version" in data
        assert "config" in data


class TestQueryEndpoint:
    """Tests for the query endpoint."""

    def test_query_empty_collection(self, test_client: TestClient) -> None:
        """Test querying an empty/nonexistent collection."""
        response = test_client.post(
            "/api/v1/query",
            json={
                "question": "What is MAVLink?",
                "top_k": 5,
                "collection": "nonexistent",
            },
        )
        # Should return 404 for nonexistent collection
        assert response.status_code == 404

    def test_query_validation(self, test_client: TestClient) -> None:
        """Test query validation."""
        # Empty question
        response = test_client.post(
            "/api/v1/query",
            json={"question": "", "top_k": 5},
        )
        assert response.status_code == 422  # Validation error


class TestIngestEndpoints:
    """Tests for the ingest endpoints."""

    def test_ingest_file(self, test_client: TestClient, tmp_path: Path) -> None:
        """Test ingesting a single file."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test document about MAVLink protocol.")

        response = test_client.post(
            "/api/v1/ingest/file",
            json={"path": str(test_file), "collection": "test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["documents_processed"] >= 1
        assert data["files_processed"] == 1

    def test_ingest_nonexistent_file(self, test_client: TestClient) -> None:
        """Test ingesting a nonexistent file."""
        response = test_client.post(
            "/api/v1/ingest/file",
            json={"path": "/nonexistent/file.txt", "collection": "test"},
        )
        assert response.status_code == 404

    def test_ingest_directory(self, test_client: TestClient, tmp_path: Path) -> None:
        """Test ingesting a directory."""
        # Create test files
        (tmp_path / "file1.txt").write_text("Document one content")
        (tmp_path / "file2.txt").write_text("Document two content")

        response = test_client.post(
            "/api/v1/ingest/directory",
            json={"path": str(tmp_path), "collection": "test", "recursive": True},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["files_processed"] >= 2

    def test_full_flow(self, test_client: TestClient, tmp_path: Path) -> None:
        """Test full ingest and query flow."""
        # Create and ingest test document
        test_file = tmp_path / "mavlink.txt"
        test_file.write_text(
            "MAVLink is a lightweight protocol for communicating with drones. "
            "It supports message authentication and encryption for secure communication."
        )

        # Ingest
        ingest_response = test_client.post(
            "/api/v1/ingest/file",
            json={"path": str(test_file), "collection": "integration_test"},
        )
        assert ingest_response.status_code == 200

        # Query
        query_response = test_client.post(
            "/api/v1/query",
            json={
                "question": "What is MAVLink?",
                "top_k": 5,
                "collection": "integration_test",
            },
        )
        assert query_response.status_code == 200

        data = query_response.json()
        assert len(data["chunks"]) > 0
        assert "MAVLink" in data["chunks"][0]["text"]


class TestCollectionEndpoints:
    """Tests for collection management endpoints."""

    def test_list_collections(self, test_client: TestClient) -> None:
        """Test listing collections."""
        response = test_client.get("/api/v1/collections")
        assert response.status_code == 200

        data = response.json()
        # Endpoint returns vector_collections and graph_collections
        assert "vector_collections" in data
        assert "vector_count" in data
        # For backward compatibility, also check if collections key exists (might be added later)
        # Or calculate total count
        total_count = data.get("vector_count", 0) + data.get("graph_count", 0)
        assert total_count >= 0

    def test_delete_collection(self, test_client: TestClient, tmp_path: Path) -> None:
        """Test deleting a collection."""
        # First create a collection by ingesting
        test_file = tmp_path / "delete_test.txt"
        test_file.write_text("Content to delete")

        test_client.post(
            "/api/v1/ingest/file",
            json={"path": str(test_file), "collection": "to_delete"},
        )

        # Delete
        response = test_client.delete("/api/v1/collections/to_delete")
        assert response.status_code == 200

        # Verify deleted
        list_response = test_client.get("/api/v1/collections")
        data = list_response.json()
        vector_collections = [c["name"] for c in data.get("vector_collections", [])]
        graph_collections = [c["name"] for c in data.get("graph_collections", [])]
        all_collections = vector_collections + graph_collections
        assert "to_delete" not in all_collections

    def test_delete_nonexistent_collection(self, test_client: TestClient) -> None:
        """Test deleting a nonexistent collection."""
        response = test_client.delete("/api/v1/collections/nonexistent_collection")
        assert response.status_code == 404
