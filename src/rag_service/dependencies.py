"""Dependency injection for the RAG service."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from rag_service.config import Settings, get_settings
from rag_service.core.chunker import DocumentChunker, create_chunker
from rag_service.core.embeddings import SentenceTransformerEmbedding, create_embedding_service
from rag_service.core.graph_extractor import EntityExtractor, create_extractor
from rag_service.core.retriever import VectorStore
from rag_service.core.router import QueryRouter, create_router
from rag_service.infrastructure.chroma_store import ChromaVectorStore
from rag_service.infrastructure.faiss_store import FAISSVectorStore
from rag_service.infrastructure.neo4j_store import (
    InMemoryGraphStore,
    Neo4jGraphStore,
    create_graph_store,
)


@lru_cache
def get_embedding_service() -> SentenceTransformerEmbedding:
    """Get the embedding service singleton with GPU safeguards."""
    settings = get_settings()
    return create_embedding_service(
        model_name=settings.embedding_model,
        device=settings.get_resolved_device(),
        batch_size=settings.embedding_batch_size,
        # GPU safeguard settings
        enable_gpu_safeguards=settings.enable_gpu_safeguards,
        max_memory_percent=settings.gpu_max_memory_percent,
        max_temperature_c=settings.gpu_max_temperature_c,
        inter_batch_delay=settings.gpu_inter_batch_delay,
        adaptive_batch_size=settings.gpu_adaptive_batch_size,
        min_batch_size=settings.gpu_min_batch_size,
        # Power management
        gpu_power_limit_watts=settings.gpu_power_limit_watts,
        enable_gpu_warmup=settings.enable_gpu_warmup,
        # Precision
        precision=settings.precision,
        # Performance
        enable_tf32=settings.enable_tf32,
        enable_cudnn_benchmark=settings.enable_cudnn_benchmark,
    )


@lru_cache
def get_vector_store() -> VectorStore:
    """Get the vector store singleton."""
    settings = get_settings()
    embedding_service = get_embedding_service()

    if settings.vector_store_backend == "faiss":
        store = FAISSVectorStore(
            persist_dir=settings.faiss_index_dir,
            embedding_dim=embedding_service.embedding_dim,
        )
        store.load()
        return store
    else:
        return ChromaVectorStore(persist_dir=settings.chroma_persist_dir)


@lru_cache
def get_chunker() -> DocumentChunker:
    """Get the document chunker singleton."""
    settings = get_settings()
    return create_chunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        pdf_strategy=settings.pdf_strategy,
    )


@lru_cache
def get_graph_store() -> Neo4jGraphStore | InMemoryGraphStore:
    """Get the graph store singleton."""
    settings = get_settings()
    if not settings.enable_graph_rag:
        return InMemoryGraphStore()

    return create_graph_store(
        backend=settings.graph_store_backend,
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )


@lru_cache
def get_query_router() -> QueryRouter:
    """Get the query router singleton."""
    settings = get_settings()
    return create_router(
        mode=settings.router_mode,
        default_strategy=settings.default_query_strategy,
        llm_model=settings.ollama_model,
    )


@lru_cache
def get_entity_extractor() -> EntityExtractor:
    """Get the entity extractor singleton."""
    settings = get_settings()
    return create_extractor(
        mode=settings.entity_extraction_mode,
        llm_model=settings.ollama_model,
        domain=settings.entity_extraction_domain,
    )


# Type aliases for cleaner endpoint signatures
SettingsDep = Annotated[Settings, Depends(get_settings)]
EmbeddingServiceDep = Annotated[SentenceTransformerEmbedding, Depends(get_embedding_service)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
ChunkerDep = Annotated[DocumentChunker, Depends(get_chunker)]
GraphStoreDep = Annotated[Neo4jGraphStore | InMemoryGraphStore, Depends(get_graph_store)]
QueryRouterDep = Annotated[QueryRouter, Depends(get_query_router)]
EntityExtractorDep = Annotated[EntityExtractor, Depends(get_entity_extractor)]


# Direct accessor functions for non-FastAPI contexts (e.g., startup, info endpoint)
def get_embedding_service_direct() -> SentenceTransformerEmbedding:
    """Get embedding service without FastAPI dependency injection."""
    return get_embedding_service()


def get_vector_store_direct() -> VectorStore:
    """Get vector store without FastAPI dependency injection."""
    return get_vector_store()
