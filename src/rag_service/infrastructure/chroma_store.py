"""ChromaDB-based vector store implementation."""

import uuid
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from numpy.typing import NDArray

from rag_service.core.exceptions import CollectionNotFoundError
from rag_service.core.retriever import Document, SearchResult, VectorStore


class ChromaVectorStore(VectorStore):
    """ChromaDB-based vector store with persistence support.

    ChromaDB provides a simpler alternative to FAISS with built-in
    metadata filtering and persistence.
    """

    def __init__(self, persist_dir: Path | str) -> None:
        """Initialize the ChromaDB vector store.

        Args:
            persist_dir: Directory for persisting the database.
        """
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        # Use new ChromaDB API (PersistentClient for local persistence)
        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
        )

    def _get_or_create_collection(self, collection: str) -> chromadb.Collection:
        """Get or create a ChromaDB collection."""
        return self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

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
        if len(documents) == 0:
            return []

        chroma_collection = self._get_or_create_collection(collection)

        # Generate IDs for documents without them
        doc_ids = []
        for doc in documents:
            if doc.document_id is None:
                doc.document_id = str(uuid.uuid4())
            doc_ids.append(doc.document_id)

        # Prepare data for ChromaDB
        texts = [doc.text for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Add to collection
        chroma_collection.add(
            ids=doc_ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,  # type: ignore[arg-type]
        )

        return doc_ids

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
        try:
            chroma_collection = self._client.get_collection(collection)
        except ValueError:
            raise CollectionNotFoundError(f"Collection '{collection}' not found")

        if chroma_collection.count() == 0:
            return []

        # Search
        k = min(top_k, chroma_collection.count())
        results = chroma_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        # Build results
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances, convert to similarity score
                # For cosine distance: similarity = 1 - distance
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1.0 - distance

                search_results.append(
                    SearchResult(
                        text=results["documents"][0][i] if results["documents"] else "",
                        metadata=dict(results["metadatas"][0][i]) if results["metadatas"] else {},
                        score=score,
                        document_id=doc_id,
                    )
                )

        return search_results

    def delete_collection(self, collection: str) -> None:
        """Delete a collection and all its documents.

        Args:
            collection: Collection name to delete.
        """
        try:
            self._client.delete_collection(collection)
        except ValueError:
            raise CollectionNotFoundError(f"Collection '{collection}' not found")

    def list_collections(self) -> list[str]:
        """List all available collections.

        Returns:
            List of collection names.
        """
        collections = self._client.list_collections()
        return [col.name for col in collections]

    def get_collection_info(self, collection: str) -> dict[str, Any]:
        """Get information about a collection.

        Args:
            collection: Collection name.

        Returns:
            Dictionary with collection information.
        """
        try:
            chroma_collection = self._client.get_collection(collection)
        except ValueError:
            raise CollectionNotFoundError(f"Collection '{collection}' not found")

        return {
            "name": collection,
            "document_count": chroma_collection.count(),
            "metadata": chroma_collection.metadata,
        }

    def persist(self) -> None:
        """Persist the database to disk.

        Note: ChromaDB PersistentClient auto-persists, so this is a no-op.
        Kept for API compatibility with FAISSVectorStore.
        """
        # ChromaDB PersistentClient automatically persists, no action needed
        pass

    def load(self) -> None:
        """Load is automatic with ChromaDB's persistent client."""
        pass
