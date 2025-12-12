"""Embedding service for generating document and query embeddings."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for multiple documents.

        Args:
            texts: List of document texts to embed.

        Returns:
            Array of embeddings with shape (n_texts, embedding_dim).
        """
        ...

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Generate embedding for a single query.

        Args:
            text: Query text to embed.

        Returns:
            Embedding array with shape (embedding_dim,).
        """
        ...

    @property
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings produced by this provider."""
        ...


class SentenceTransformerEmbedding:
    """Embedding service using sentence-transformers library.

    Supports CUDA (Windows/Linux), MPS (macOS), and CPU fallback.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        """Initialize the embedding service.

        Args:
            model_name: HuggingFace model ID for embeddings.
            device: Device to run inference on ("auto", "cpu", "cuda", "mps").
            batch_size: Batch size for embedding generation.
        """
        from sentence_transformers import SentenceTransformer

        resolved_device = self._resolve_device(device)
        self._model = SentenceTransformer(model_name, device=resolved_device)
        self._batch_size = batch_size
        self._model_name = model_name

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve device string to actual device.

        Args:
            device: Device specification ("auto", "cpu", "cuda", "mps").

        Returns:
            Resolved device string.
        """
        if device != "auto":
            return device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for multiple documents.

        Args:
            texts: List of document texts to embed.

        Returns:
            Array of embeddings with shape (n_texts, embedding_dim).
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.embedding_dim)

        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Generate embedding for a single query.

        Args:
            text: Query text to embed.

        Returns:
            Embedding array with shape (embedding_dim,).
        """
        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings produced by this model."""
        return self._model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    def get_device_info(self) -> dict[str, str]:
        """Get information about the device being used.

        Returns:
            Dictionary with device information.
        """
        device = str(self._model.device)
        info = {"device": device, "model": self._model_name}

        try:
            import torch

            if device == "cuda" and torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
        except Exception:
            pass

        return info


def create_embedding_service(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "auto",
    batch_size: int = 32,
) -> SentenceTransformerEmbedding:
    """Factory function to create an embedding service.

    Args:
        model_name: HuggingFace model ID for embeddings.
        device: Device to run inference on.
        batch_size: Batch size for embedding generation.

    Returns:
        Configured embedding service instance.
    """
    return SentenceTransformerEmbedding(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )

