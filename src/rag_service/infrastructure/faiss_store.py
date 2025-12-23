"""FAISS-based vector store implementation."""

import json
import pickle
import uuid
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from numpy.typing import NDArray

from rag_service.core.exceptions import CollectionNotFoundError
from rag_service.core.retriever import Document, SearchResult, VectorStore


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store with persistence support.

    Uses FAISS IndexFlatIP for inner product similarity search.
    Stores metadata separately in a pickle file.
    """

    def __init__(self, persist_dir: Path | str, embedding_dim: int = 384) -> None:
        """Initialize the FAISS vector store.

        Args:
            persist_dir: Directory for persisting indices.
            embedding_dim: Dimension of embedding vectors.
        """
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_dim = embedding_dim

        # In-memory storage: collection_name -> (index, documents, ids)
        self._indices: dict[str, faiss.IndexFlatIP] = {}
        self._documents: dict[str, list[Document]] = {}
        self._id_map: dict[str, list[str]] = {}

    def _get_or_create_collection(self, collection: str) -> None:
        """Ensure a collection exists."""
        if collection not in self._indices:
            self._indices[collection] = faiss.IndexFlatIP(self._embedding_dim)
            self._documents[collection] = []
            self._id_map[collection] = []

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

        self._get_or_create_collection(collection)

        # Generate IDs for documents without them
        doc_ids = []
        for doc in documents:
            if doc.document_id is None:
                doc.document_id = str(uuid.uuid4())
            doc_ids.append(doc.document_id)

        # Normalize embeddings for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        # Add to FAISS index
        self._indices[collection].add(embeddings)

        # Store documents and IDs
        self._documents[collection].extend(documents)
        self._id_map[collection].extend(doc_ids)

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
        if collection not in self._indices:
            raise CollectionNotFoundError(f"Collection '{collection}' not found")

        index = self._indices[collection]
        if index.ntotal == 0:
            return []

        # Ensure query is 2D and normalized
        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)

        # Search
        k = min(top_k, index.ntotal)
        scores, indices = index.search(query, k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            doc = self._documents[collection][idx]
            results.append(
                SearchResult(
                    text=doc.text,
                    metadata=doc.metadata,
                    score=float(score),
                    document_id=self._id_map[collection][idx],
                )
            )

        return results

    def delete_collection(self, collection: str) -> None:
        """Delete a collection and all its documents.

        Args:
            collection: Collection name to delete.
        """
        if collection not in self._indices:
            raise CollectionNotFoundError(f"Collection '{collection}' not found")

        del self._indices[collection]
        del self._documents[collection]
        del self._id_map[collection]

        # Remove persisted files
        index_path = self._persist_dir / f"{collection}.faiss"
        meta_path = self._persist_dir / f"{collection}.meta"
        if index_path.exists():
            index_path.unlink()
        if meta_path.exists():
            meta_path.unlink()

    def list_collections(self) -> list[str]:
        """List all available collections.

        Returns:
            List of collection names.
        """
        # Include both in-memory and persisted collections
        collections = set(self._indices.keys())

        # Check for persisted collections not yet loaded
        for path in self._persist_dir.glob("*.faiss"):
            collections.add(path.stem)

        return sorted(collections)

    def get_collection_info(self, collection: str) -> dict[str, Any]:
        """Get information about a collection.

        Args:
            collection: Collection name.

        Returns:
            Dictionary with collection information.
        """
        # Load if not in memory
        if collection not in self._indices:
            self._load_collection(collection)

        if collection not in self._indices:
            raise CollectionNotFoundError(f"Collection '{collection}' not found")

        index = self._indices[collection]
        return {
            "name": collection,
            "document_count": index.ntotal,
            "embedding_dim": self._embedding_dim,
        }

    def persist(self) -> None:
        """Persist all collections to disk."""
        for collection in self._indices:
            self._persist_collection(collection)

    def _persist_collection(self, collection: str) -> None:
        """Persist a single collection to disk."""
        if collection not in self._indices:
            return

        index_path = self._persist_dir / f"{collection}.faiss"
        meta_path = self._persist_dir / f"{collection}.meta"

        # Save FAISS index
        faiss.write_index(self._indices[collection], str(index_path))

        # Save metadata
        meta = {
            "documents": [
                {
                    "text": doc.text,
                    "metadata": doc.metadata,
                    "document_id": doc.document_id,
                }
                for doc in self._documents[collection]
            ],
            "ids": self._id_map[collection],
            "embedding_dim": self._embedding_dim,
        }
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

    def load(self) -> None:
        """Load all persisted collections from disk."""
        for path in self._persist_dir.glob("*.faiss"):
            collection = path.stem
            if collection not in self._indices:
                self._load_collection(collection)

    def _load_collection(self, collection: str) -> None:
        """Load a single collection from disk."""
        index_path = self._persist_dir / f"{collection}.faiss"
        meta_path = self._persist_dir / f"{collection}.meta"

        if not index_path.exists() or not meta_path.exists():
            return

        # Load FAISS index
        self._indices[collection] = faiss.read_index(str(index_path))

        # Load metadata
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self._documents[collection] = [
            Document(
                text=doc["text"],
                metadata=doc["metadata"],
                document_id=doc["document_id"],
            )
            for doc in meta["documents"]
        ]
        self._id_map[collection] = meta["ids"]
        self._embedding_dim = meta.get("embedding_dim", self._embedding_dim)

