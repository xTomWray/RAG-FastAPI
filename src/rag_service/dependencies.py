"""Dependency injection for the RAG service."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from rag_service.config import Settings, get_settings
from rag_service.core.chunker import DocumentChunker, create_chunker
from rag_service.core.embeddings import SentenceTransformerEmbedding, create_embedding_service
from rag_service.core.retriever import VectorStore
from rag_service.infrastructure.chroma_store import ChromaVectorStore
from rag_service.infrastructure.faiss_store import FAISSVectorStore


@lru_cache
def get_embedding_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> SentenceTransformerEmbedding:
    """Get the embedding service singleton.

    Args:
        settings: Application settings.

    Returns:
        Configured embedding service.
    """
    return create_embedding_service(
        model_name=settings.embedding_model,
        device=settings.get_resolved_device(),
        batch_size=settings.embedding_batch_size,
    )


@lru_cache
def get_vector_store(
    settings: Annotated[Settings, Depends(get_settings)],
) -> VectorStore:
    """Get the vector store singleton.

    Args:
        settings: Application settings.

    Returns:
        Configured vector store.
    """
    # Need embedding service to get dimension
    embedding_service = get_embedding_service(settings)

    if settings.vector_store_backend == "faiss":
        store = FAISSVectorStore(
            persist_dir=settings.faiss_index_dir,
            embedding_dim=embedding_service.embedding_dim,
        )
        # Load existing indices
        store.load()
        return store
    else:
        store = ChromaVectorStore(persist_dir=settings.chroma_persist_dir)
        return store


@lru_cache
def get_chunker(
    settings: Annotated[Settings, Depends(get_settings)],
) -> DocumentChunker:
    """Get the document chunker singleton.

    Args:
        settings: Application settings.

    Returns:
        Configured document chunker.
    """
    return create_chunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        pdf_strategy=settings.pdf_strategy,
    )


# Type aliases for cleaner endpoint signatures
SettingsDep = Annotated[Settings, Depends(get_settings)]
EmbeddingServiceDep = Annotated[SentenceTransformerEmbedding, Depends(get_embedding_service)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
ChunkerDep = Annotated[DocumentChunker, Depends(get_chunker)]

