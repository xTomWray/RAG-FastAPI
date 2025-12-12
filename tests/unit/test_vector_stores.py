"""Unit tests for vector store implementations."""

from pathlib import Path

import numpy as np
import pytest

from rag_service.core.exceptions import CollectionNotFoundError
from rag_service.core.retriever import Document
from rag_service.infrastructure.faiss_store import FAISSVectorStore


class TestFAISSVectorStore:
    """Tests for FAISS vector store."""

    @pytest.fixture
    def store(self, temp_dir: Path) -> FAISSVectorStore:
        """Create a FAISS store for testing."""
        return FAISSVectorStore(persist_dir=temp_dir, embedding_dim=384)

    def test_add_and_search_documents(
        self,
        store: FAISSVectorStore,
        sample_documents: list[Document],
        sample_embeddings: np.ndarray,
    ) -> None:
        """Test adding and searching documents."""
        # Add documents
        doc_ids = store.add_documents(
            documents=sample_documents,
            embeddings=sample_embeddings,
            collection="test",
        )

        assert len(doc_ids) == 3
        assert all(isinstance(id, str) for id in doc_ids)

        # Search
        query_embedding = sample_embeddings[0]  # Use first doc's embedding as query
        results = store.search(query_embedding, top_k=2, collection="test")

        assert len(results) == 2
        assert results[0].score > results[1].score  # Sorted by similarity

    def test_search_nonexistent_collection(
        self,
        store: FAISSVectorStore,
        sample_embeddings: np.ndarray,
    ) -> None:
        """Test searching a nonexistent collection."""
        with pytest.raises(CollectionNotFoundError):
            store.search(sample_embeddings[0], collection="nonexistent")

    def test_list_collections(
        self,
        store: FAISSVectorStore,
        sample_documents: list[Document],
        sample_embeddings: np.ndarray,
    ) -> None:
        """Test listing collections."""
        store.add_documents(sample_documents, sample_embeddings, collection="col1")
        store.add_documents(sample_documents, sample_embeddings, collection="col2")

        collections = store.list_collections()
        assert "col1" in collections
        assert "col2" in collections

    def test_delete_collection(
        self,
        store: FAISSVectorStore,
        sample_documents: list[Document],
        sample_embeddings: np.ndarray,
    ) -> None:
        """Test deleting a collection."""
        store.add_documents(sample_documents, sample_embeddings, collection="to_delete")
        assert "to_delete" in store.list_collections()

        store.delete_collection("to_delete")
        assert "to_delete" not in store.list_collections()

    def test_delete_nonexistent_collection(self, store: FAISSVectorStore) -> None:
        """Test deleting a nonexistent collection."""
        with pytest.raises(CollectionNotFoundError):
            store.delete_collection("nonexistent")

    def test_get_collection_info(
        self,
        store: FAISSVectorStore,
        sample_documents: list[Document],
        sample_embeddings: np.ndarray,
    ) -> None:
        """Test getting collection info."""
        store.add_documents(sample_documents, sample_embeddings, collection="info_test")

        info = store.get_collection_info("info_test")
        assert info["name"] == "info_test"
        assert info["document_count"] == 3
        assert info["embedding_dim"] == 384

    def test_persist_and_load(
        self,
        temp_dir: Path,
        sample_documents: list[Document],
        sample_embeddings: np.ndarray,
    ) -> None:
        """Test persisting and loading the store."""
        # Create and populate store
        store1 = FAISSVectorStore(persist_dir=temp_dir, embedding_dim=384)
        store1.add_documents(sample_documents, sample_embeddings, collection="persist_test")
        store1.persist()

        # Create new store and load
        store2 = FAISSVectorStore(persist_dir=temp_dir, embedding_dim=384)
        store2.load()

        # Verify data is loaded
        assert "persist_test" in store2.list_collections()
        info = store2.get_collection_info("persist_test")
        assert info["document_count"] == 3

    def test_add_empty_documents(self, store: FAISSVectorStore) -> None:
        """Test adding empty document list."""
        doc_ids = store.add_documents(
            documents=[],
            embeddings=np.array([]).reshape(0, 384).astype(np.float32),
            collection="empty",
        )
        assert len(doc_ids) == 0

    def test_search_empty_collection(
        self,
        store: FAISSVectorStore,
        sample_embeddings: np.ndarray,
    ) -> None:
        """Test searching an empty collection."""
        store._get_or_create_collection("empty")
        results = store.search(sample_embeddings[0], collection="empty")
        assert len(results) == 0

    def test_top_k_larger_than_collection(
        self,
        store: FAISSVectorStore,
        sample_documents: list[Document],
        sample_embeddings: np.ndarray,
    ) -> None:
        """Test requesting more results than available."""
        store.add_documents(sample_documents, sample_embeddings, collection="small")

        # Request 10 results from collection with 3 documents
        results = store.search(sample_embeddings[0], top_k=10, collection="small")
        assert len(results) == 3


class TestChromaVectorStore:
    """Tests for ChromaDB vector store.

    Note: ChromaDB tests may be slower due to database initialization.
    """

    @pytest.fixture
    def store(self, temp_dir: Path):
        """Create a ChromaDB store for testing."""
        from rag_service.infrastructure.chroma_store import ChromaVectorStore

        return ChromaVectorStore(persist_dir=temp_dir)

    def test_add_and_search_documents(
        self,
        store,
        sample_documents: list[Document],
        sample_embeddings: np.ndarray,
    ) -> None:
        """Test adding and searching documents."""
        doc_ids = store.add_documents(
            documents=sample_documents,
            embeddings=sample_embeddings,
            collection="test",
        )

        assert len(doc_ids) == 3

        results = store.search(sample_embeddings[0], top_k=2, collection="test")
        assert len(results) == 2

    def test_list_collections(
        self,
        store,
        sample_documents: list[Document],
        sample_embeddings: np.ndarray,
    ) -> None:
        """Test listing collections."""
        store.add_documents(sample_documents, sample_embeddings, collection="chroma1")
        store.add_documents(sample_documents, sample_embeddings, collection="chroma2")

        collections = store.list_collections()
        assert "chroma1" in collections
        assert "chroma2" in collections

