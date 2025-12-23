"""Unit tests for the embedding service."""

import numpy as np
import pytest

from rag_service.core.embeddings import SentenceTransformerEmbedding, create_embedding_service


class TestSentenceTransformerEmbedding:
    """Tests for SentenceTransformerEmbedding class.

    Note: These tests require the sentence-transformers library and will
    download models on first run. Use pytest markers to skip in CI if needed.
    """

    @pytest.fixture(scope="class")
    def embedding_service(self) -> SentenceTransformerEmbedding:
        """Create embedding service with small model for testing."""
        return create_embedding_service(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            batch_size=8,
        )

    def test_embed_single_query(
        self,
        embedding_service: SentenceTransformerEmbedding,
    ) -> None:
        """Test embedding a single query."""
        query = "What is MAVLink?"
        embedding = embedding_service.embed_query(query)

        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert len(embedding.shape) == 1
        assert embedding.shape[0] == embedding_service.embedding_dim

    def test_embed_documents(
        self,
        embedding_service: SentenceTransformerEmbedding,
    ) -> None:
        """Test embedding multiple documents."""
        documents = [
            "MAVLink is a protocol.",
            "It is used for drones.",
            "Security is important.",
        ]
        embeddings = embedding_service.embed_documents(documents)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.dtype == np.float32
        assert embeddings.shape == (3, embedding_service.embedding_dim)

    def test_embed_empty_documents(
        self,
        embedding_service: SentenceTransformerEmbedding,
    ) -> None:
        """Test embedding empty document list."""
        embeddings = embedding_service.embed_documents([])

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (0, embedding_service.embedding_dim)

    def test_embeddings_normalized(
        self,
        embedding_service: SentenceTransformerEmbedding,
    ) -> None:
        """Test that embeddings are normalized."""
        embedding = embedding_service.embed_query("Test query")
        norm = np.linalg.norm(embedding)

        # Should be approximately 1.0 (normalized)
        assert abs(norm - 1.0) < 1e-5

    def test_similar_texts_have_similar_embeddings(
        self,
        embedding_service: SentenceTransformerEmbedding,
    ) -> None:
        """Test that similar texts produce similar embeddings."""
        text1 = "The cat sat on the mat."
        text2 = "A cat was sitting on the mat."
        text3 = "Quantum physics explains subatomic particles."

        emb1 = embedding_service.embed_query(text1)
        emb2 = embedding_service.embed_query(text2)
        emb3 = embedding_service.embed_query(text3)

        # Cosine similarity via dot product (embeddings are normalized)
        sim_1_2 = np.dot(emb1, emb2)
        sim_1_3 = np.dot(emb1, emb3)

        # Similar texts should have higher similarity
        assert sim_1_2 > sim_1_3

    def test_embedding_dim_property(
        self,
        embedding_service: SentenceTransformerEmbedding,
    ) -> None:
        """Test embedding dimension property."""
        # MiniLM-L6-v2 has 384 dimensions
        assert embedding_service.embedding_dim == 384

    def test_model_name_property(
        self,
        embedding_service: SentenceTransformerEmbedding,
    ) -> None:
        """Test model name property."""
        assert "MiniLM" in embedding_service.model_name

    def test_device_info(
        self,
        embedding_service: SentenceTransformerEmbedding,
    ) -> None:
        """Test device info method."""
        info = embedding_service.get_device_info()

        assert "device" in info
        assert "model" in info
        assert info["device"] == "cpu"

    def test_resolve_device_cpu(self) -> None:
        """Test device resolution for CPU."""
        device = SentenceTransformerEmbedding._resolve_device("cpu")
        assert device == "cpu"

    def test_resolve_device_auto(self) -> None:
        """Test auto device resolution."""
        device = SentenceTransformerEmbedding._resolve_device("auto")
        # Should return one of: cuda, mps, or cpu
        assert device in ["cuda", "mps", "cpu"]

