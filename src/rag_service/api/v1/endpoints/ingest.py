"""Ingest endpoints for document processing and indexing."""

import contextlib
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from rag_service.api.v1.schemas import (
    DirectoryIngestRequest,
    FileIngestRequest,
    IngestResponse,
    IngestStatus,
)
from rag_service.config import get_settings
from rag_service.core.crash_logger import get_crash_logger
from rag_service.core.exceptions import (
    CollectionNotFoundError,
    DocumentProcessingError,
    UnsupportedFileTypeError,
)
from rag_service.core.stats import get_stats_collector
from rag_service.dependencies import (
    get_chunker,
    get_embedding_service,
    get_entity_extractor,
    get_graph_store,
    get_vector_store,
)

logger = logging.getLogger(__name__)
crash_log = get_crash_logger()
router = APIRouter(tags=["ingest"])


def _build_knowledge_graph(
    documents: list[Any],
    collection: str,
) -> dict[str, Any]:
    """Extract entities and relationships from documents and add to graph store.

    Args:
        documents: List of processed documents.
        collection: Collection name.

    Returns:
        Dictionary with entity and relationship counts.
    """
    settings = get_settings()
    if not settings.enable_graph_rag:
        return {"entities": 0, "relationships": 0}

    try:
        extractor = get_entity_extractor()
        graph_store = get_graph_store()

        all_entities = []
        all_relationships = []

        # Extract from each document
        for doc in documents:
            entities, relationships = extractor.extract(
                text=doc.text,
                source=doc.metadata.get("source"),
            )
            all_entities.extend(entities)
            all_relationships.extend(relationships)

        # Deduplicate entities
        seen_entities = set()
        unique_entities = []
        for entity in all_entities:
            if entity.name not in seen_entities:
                seen_entities.add(entity.name)
                unique_entities.append(entity)

        # Deduplicate relationships
        seen_rels = set()
        unique_relationships = []
        for rel in all_relationships:
            key = (rel.source, rel.target, rel.relationship_type)
            if key not in seen_rels:
                seen_rels.add(key)
                unique_relationships.append(rel)

        # Add to graph store
        if unique_entities:
            graph_store.add_entities(unique_entities, collection=collection)
            logger.info(f"Added {len(unique_entities)} entities to graph store")

        if unique_relationships:
            graph_store.add_relationships(unique_relationships, collection=collection)
            logger.info(f"Added {len(unique_relationships)} relationships to graph store")

        return {
            "entities": len(unique_entities),
            "relationships": len(unique_relationships),
        }

    except Exception as e:
        logger.warning(f"Failed to build knowledge graph: {e}")
        return {"entities": 0, "relationships": 0, "error": str(e)}


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(request: FileIngestRequest) -> IngestResponse:
    """Ingest a single file into the vector store and knowledge graph.

    Processes the file, chunks it, generates embeddings, extracts entities
    and relationships, and stores everything in the specified collection.

    Args:
        request: File ingest request with path and collection.

    Returns:
        Ingest response with processing results.

    Raises:
        HTTPException: If file not found or processing fails.
    """
    try:
        # Get services
        chunker = get_chunker()
        embedding_service = get_embedding_service()
        vector_store = get_vector_store()

        # Process the file
        file_path = Path(request.path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {request.path}")
        documents = chunker.process_file(file_path)

        if not documents:
            return IngestResponse(
                status=IngestStatus.SUCCESS,
                documents_processed=0,
                files_processed=1,
                collection=request.collection,
                errors=["No content extracted from file"],
            )

        # Generate embeddings
        logger.info(f"[INGEST] Generating embeddings for {len(documents)} chunks")
        texts = [doc.text for doc in documents]
        embeddings = embedding_service.embed_documents(texts)
        logger.info("[INGEST] Embeddings complete, adding to vector store")

        # Add to vector store
        vector_store.add_documents(
            documents=documents,
            embeddings=embeddings,
            collection=request.collection,
        )
        logger.info("[INGEST] Added to vector store, persisting...")

        # Persist vector store
        vector_store.persist()
        logger.info("[INGEST] Persisted, building knowledge graph...")

        # Build knowledge graph (if enabled)
        _build_knowledge_graph(documents, request.collection)
        logger.info("[INGEST] Complete, returning response")

        return IngestResponse(
            status=IngestStatus.SUCCESS,
            documents_processed=len(documents),
            files_processed=1,
            collection=request.collection,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DocumentProcessingError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")


@router.post("/ingest/directory", response_model=IngestResponse)
async def ingest_directory(request: DirectoryIngestRequest) -> IngestResponse:
    """Ingest all supported files in a directory.

    Recursively processes files in the directory, chunks them,
    generates embeddings, extracts entities/relationships, and stores
    in the specified collection.

    Args:
        request: Directory ingest request with path and options.

    Returns:
        Ingest response with processing results.

    Raises:
        HTTPException: If directory not found or processing fails.
    """
    start_time = time.perf_counter()
    success = True
    documents_count = 0
    chunks_count = 0
    bytes_processed = 0

    # Start crash-safe logging for entire ingest operation
    crash_log.info(
        "INGEST_DIRECTORY_START",
        capture_state=True,
        path=request.path,
        collection=request.collection,
        recursive=request.recursive,
    )

    try:
        # Get services
        chunker = get_chunker()
        embedding_service = get_embedding_service()
        vector_store = get_vector_store()

        # Process the directory
        dir_path = Path(request.path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {request.path}")

        # STEP 1: Chunking
        crash_log.info(
            "INGEST_STEP_CHUNKING",
            capture_state=True,
            directory=str(dir_path),
        )
        documents = chunker.process_directory(dir_path, recursive=request.recursive)

        if not documents:
            crash_log.info("INGEST_NO_DOCUMENTS", path=request.path)
            return IngestResponse(
                status=IngestStatus.SUCCESS,
                documents_processed=0,
                files_processed=0,
                collection=request.collection,
                errors=["No documents found in directory"],
            )

        # Count unique files and bytes
        files_processed = len({doc.metadata.get("source") for doc in documents})
        documents_count = files_processed
        chunks_count = len(documents)
        bytes_processed = sum(len(doc.text.encode("utf-8")) for doc in documents)

        crash_log.info(
            "INGEST_CHUNKING_COMPLETE",
            capture_state=True,
            total_chunks=len(documents),
            files_processed=files_processed,
        )

        # STEP 2: Generate embeddings (GPU-intensive)
        crash_log.info(
            "INGEST_STEP_EMBEDDING",
            capture_state=True,
            num_chunks=len(documents),
        )
        texts = [doc.text for doc in documents]
        embeddings = embedding_service.embed_documents(texts)
        crash_log.info(
            "INGEST_EMBEDDING_COMPLETE",
            capture_state=True,
            embeddings_shape=str(embeddings.shape),
        )

        # STEP 3: Add to vector store
        crash_log.info(
            "INGEST_STEP_VECTOR_STORE",
            capture_state=True,
            collection=request.collection,
        )
        vector_store.add_documents(
            documents=documents,
            embeddings=embeddings,
            collection=request.collection,
        )

        # Persist vector store (ChromaDB auto-persists, FAISS needs explicit persist)
        if hasattr(vector_store, "persist"):
            with contextlib.suppress(AttributeError):
                vector_store.persist()

        crash_log.info("INGEST_VECTOR_STORE_COMPLETE", capture_state=True)

        # STEP 4: Build knowledge graph (if enabled)
        crash_log.info("INGEST_STEP_GRAPH", capture_state=True)
        graph_stats = _build_knowledge_graph(documents, request.collection)
        crash_log.info(
            "INGEST_GRAPH_COMPLETE",
            capture_state=True,
            entities=graph_stats.get("entities", 0),
            relationships=graph_stats.get("relationships", 0),
        )

        crash_log.info(
            "INGEST_DIRECTORY_SUCCESS",
            capture_state=True,
            documents_processed=len(documents),
            files_processed=files_processed,
        )

        return IngestResponse(
            status=IngestStatus.SUCCESS,
            documents_processed=len(documents),
            files_processed=files_processed,
            collection=request.collection,
        )

    except FileNotFoundError as e:
        success = False
        crash_log.error("INGEST_FILE_NOT_FOUND", error=str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except DocumentProcessingError as e:
        success = False
        crash_log.error("INGEST_PROCESSING_ERROR", exc_info=True, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        success = False
        crash_log.critical("INGEST_UNEXPECTED_ERROR", exc_info=True, error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")
    finally:
        # Record ingestion stats
        duration_ms = (time.perf_counter() - start_time) * 1000
        stats = get_stats_collector()
        stats.record_ingestion(
            duration_ms=duration_ms,
            documents=documents_count,
            chunks=chunks_count,
            bytes_processed=bytes_processed,
            file_type="directory",
            success=success,
        )


@router.get("/collections")
async def list_collections() -> dict[str, Any]:
    """List all available collections.

    Returns:
        Dictionary with collection names and counts from both stores.
    """
    settings = get_settings()

    try:
        vector_store = get_vector_store()
        vector_collections = vector_store.list_collections()

        collection_info = []
        for name in vector_collections:
            try:
                info = vector_store.get_collection_info(name)
                collection_info.append(info)
            except Exception:
                collection_info.append({"name": name, "error": "Could not get info"})

        result = {
            "vector_collections": collection_info,
            "vector_count": len(vector_collections),
        }

        # Add graph store info if enabled
        if settings.enable_graph_rag:
            try:
                graph_store = get_graph_store()
                graph_collections = graph_store.list_collections()

                graph_info = []
                for name in graph_collections:
                    try:
                        stats = graph_store.get_collection_stats(name)
                        graph_info.append({"name": name, **stats})
                    except Exception:
                        graph_info.append({"name": name, "error": "Could not get stats"})

                result["graph_collections"] = graph_info
                result["graph_count"] = len(graph_collections)
            except Exception as e:
                result["graph_error"] = str(e)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {e}")


@router.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str) -> dict[str, str]:
    """Delete a collection and all its documents from both stores.

    Args:
        collection_name: Name of the collection to delete.

    Returns:
        Success message.

    Raises:
        HTTPException: If collection not found.
    """
    settings = get_settings()
    deleted = []

    try:
        # Delete from vector store
        vector_store = get_vector_store()
        try:
            vector_store.delete_collection(collection_name)
            # Persist if method exists (FAISS needs it, ChromaDB auto-persists)
            if hasattr(vector_store, "persist"):
                with contextlib.suppress(AttributeError):
                    vector_store.persist()
            deleted.append("vector")
        except (CollectionNotFoundError, Exception) as e:
            # ChromaDB raises different exception types, catch all and check if it's "not found"
            error_msg = str(e).lower()
            if "not found" in error_msg or "does not exist" in error_msg:
                pass  # Collection doesn't exist, which is fine
            else:
                # Re-raise if it's a different error
                raise

        # Delete from graph store if enabled
        if settings.enable_graph_rag:
            try:
                graph_store = get_graph_store()
                # Check if collection exists in graph store before deleting
                graph_collections = graph_store.list_collections()
                if collection_name in graph_collections:
                    graph_store.delete_collection(collection_name)
                    deleted.append("graph")
            except Exception:
                pass

        if not deleted:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

        return {"message": f"Collection '{collection_name}' deleted from: {', '.join(deleted)}"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {e}")


@router.get("/graph/stats/{collection_name}")
async def get_graph_stats(collection_name: str) -> dict[str, Any]:
    """Get statistics about a collection's knowledge graph.

    Args:
        collection_name: Name of the collection.

    Returns:
        Graph statistics including node and relationship counts.
    """
    settings = get_settings()

    if not settings.enable_graph_rag:
        raise HTTPException(status_code=400, detail="GraphRAG is not enabled")

    try:
        graph_store = get_graph_store()
        stats = graph_store.get_collection_stats(collection_name)
        return {"collection": collection_name, **stats}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get graph stats: {e}")
