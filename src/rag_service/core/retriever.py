"""Abstract retriever interface for vector stores."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class SearchResult:
    """Result from a vector similarity search."""

    text: str
    metadata: dict[str, Any]
    score: float
    document_id: str


@dataclass
class Document:
    """A document with text content and metadata."""

    text: str
    metadata: dict[str, Any]
    document_id: str | None = None


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add_documents(
        self,
        documents: list[Document],
        embeddings: NDArray[np.float32],
        collection: str = "default",
    ) -> list[str]:
        """Add documents with their embeddings to the store.

        Args:
            documents: List of documents to add.
            embeddings: Embeddings array with shape (n_docs, embedding_dim).
            collection: Collection name to add documents to.

        Returns:
            List of document IDs.
        """
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: NDArray[np.float32],
        top_k: int = 5,
        collection: str = "default",
    ) -> list[SearchResult]:
        """Search for similar documents.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            collection: Collection to search in.

        Returns:
            List of search results sorted by similarity.
        """
        ...

    @abstractmethod
    def delete_collection(self, collection: str) -> None:
        """Delete a collection and all its documents.

        Args:
            collection: Collection name to delete.
        """
        ...

    @abstractmethod
    def list_collections(self) -> list[str]:
        """List all available collections.

        Returns:
            List of collection names.
        """
        ...

    @abstractmethod
    def get_collection_info(self, collection: str) -> dict[str, Any]:
        """Get information about a collection.

        Args:
            collection: Collection name.

        Returns:
            Dictionary with collection information.
        """
        ...

    @abstractmethod
    def persist(self) -> None:
        """Persist the vector store to disk."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load the vector store from disk."""
        ...
