"""Unit tests for cross-encoder reranker service."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestCrossEncoderReranker:
    """Tests for the CrossEncoderReranker class."""

    @pytest.fixture
    def mock_cross_encoder(self):
        """Create a mock CrossEncoder model."""
        mock_model = MagicMock()
        # Simulate predict returning scores
        mock_model.predict.return_value = np.array([0.9, 0.7, 0.3, 0.8, 0.5])
        return mock_model

    @pytest.fixture
    def reranker(self, mock_cross_encoder):
        """Create a reranker with mocked model."""
        with (
            patch("sentence_transformers.CrossEncoder", return_value=mock_cross_encoder),
            patch("rag_service.core.reranker.get_diagnostic_logger") as mock_diag,
        ):
            mock_diag.return_value = MagicMock()
            from rag_service.core.reranker import CrossEncoderReranker

            # Create reranker with CPU to avoid GPU initialization
            reranker = CrossEncoderReranker(
                model_name="mock-model",
                device="cpu",
                enable_gpu_safeguards=False,
            )
            return reranker

    def test_rerank_returns_sorted_results(self, reranker):
        """Test that rerank returns results sorted by score descending."""
        query = "What is MAVLink authentication?"
        documents = [
            "Doc about unrelated topic",
            "Doc about MAVLink basics",
            "Doc about authentication in general",
            "Doc about MAVLink authentication specifically",
            "Doc about protocols",
        ]

        results = reranker.rerank(query, documents)

        # Results should be sorted by score descending
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_returns_original_indices(self, reranker):
        """Test that rerank returns correct original indices."""
        query = "test query"
        documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]

        results = reranker.rerank(query, documents)

        # Should return all indices
        indices = {idx for idx, _ in results}
        assert indices == {0, 1, 2, 3, 4}

    def test_rerank_with_top_k(self, reranker):
        """Test that top_k limits results."""
        query = "test query"
        documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]

        results = reranker.rerank(query, documents, top_k=3)

        assert len(results) == 3
        # Results should still be sorted by score (top 3 highest)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_empty_documents(self, reranker):
        """Test handling of empty document list."""
        results = reranker.rerank("query", [])
        assert results == []

    def test_score_pairs_returns_array(self, reranker):
        """Test that score_pairs returns numpy array."""
        query = "test"
        documents = ["doc1", "doc2"]

        # Update mock for smaller input
        reranker._model.predict.return_value = np.array([0.8, 0.6])

        scores = reranker.score_pairs(query, documents)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == 2
        assert scores.dtype == np.float32

    def test_score_pairs_empty_returns_empty_array(self, reranker):
        """Test that empty documents returns empty array."""
        scores = reranker.score_pairs("query", [])
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 0

    def test_model_name_property(self, reranker):
        """Test model_name property."""
        assert reranker.model_name == "mock-model"

    def test_get_device_info(self, reranker):
        """Test device info returns expected keys."""
        info = reranker.get_device_info()

        assert "device" in info
        assert "model" in info
        assert "precision" in info
        assert info["device"] == "cpu"


class TestRerankerFactory:
    """Tests for the create_reranker factory function."""

    def test_create_reranker_with_defaults(self):
        """Test factory function with default parameters."""
        with (
            patch("sentence_transformers.CrossEncoder") as mock_ce,
            patch("rag_service.core.reranker.get_diagnostic_logger") as mock_diag,
        ):
            mock_diag.return_value = MagicMock()
            mock_ce.return_value = MagicMock()

            from rag_service.core.reranker import create_reranker

            reranker = create_reranker(device="cpu")

            assert reranker is not None
            assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_create_reranker_with_custom_model(self):
        """Test factory function with custom model."""
        with (
            patch("sentence_transformers.CrossEncoder") as mock_ce,
            patch("rag_service.core.reranker.get_diagnostic_logger") as mock_diag,
        ):
            mock_diag.return_value = MagicMock()
            mock_ce.return_value = MagicMock()

            from rag_service.core.reranker import create_reranker

            reranker = create_reranker(
                model_name="BAAI/bge-reranker-base",
                device="cpu",
                batch_size=16,
            )

            assert reranker.model_name == "BAAI/bge-reranker-base"


class TestRerankStats:
    """Tests for rerank statistics recording."""

    def test_stats_recorded_on_rerank(self):
        """Test that stats are recorded after reranking."""
        with (
            patch("sentence_transformers.CrossEncoder") as mock_ce,
            patch("rag_service.core.reranker.get_diagnostic_logger") as mock_diag,
            patch("rag_service.core.reranker.get_stats_collector") as mock_stats,
        ):
            mock_diag.return_value = MagicMock()
            mock_ce.return_value = MagicMock()
            mock_ce.return_value.predict.return_value = np.array([0.5, 0.8])
            mock_collector = MagicMock()
            mock_stats.return_value = mock_collector

            from rag_service.core.reranker import CrossEncoderReranker

            reranker = CrossEncoderReranker(device="cpu", enable_gpu_safeguards=False)
            reranker.score_pairs("query", ["doc1", "doc2"])

            mock_collector.record_rerank.assert_called_once()
            call_kwargs = mock_collector.record_rerank.call_args.kwargs
            assert call_kwargs["num_documents"] == 2
            assert call_kwargs["success"] is True
