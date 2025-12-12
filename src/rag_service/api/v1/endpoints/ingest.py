"""Ingest endpoints for document processing and indexing."""

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
from rag_service.core.exceptions import (
    CollectionNotFoundError,
    DocumentProcessingError,
    UnsupportedFileTypeError,
)
from rag_service.dependencies import get_chunker, get_embedding_service, get_vector_store

router = APIRouter(tags=["ingest"])


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(request: FileIngestRequest) -> IngestResponse:
    """Ingest a single file into the vector store.

    Processes the file, chunks it, generates embeddings, and stores
    the chunks in the specified collection.

    Args:
        request: File ingest request with path and collection.

    Returns:
        Ingest response with processing results.

    Raises:
        HTTPException: If file not found or processing fails.
    """
    settings = get_settings()

    try:
        # Get services
        chunker = get_chunker(settings)
        embedding_service = get_embedding_service(settings)
        vector_store = get_vector_store(settings)

        # Process the file
        file_path = Path(request.path)
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
        texts = [doc.text for doc in documents]
        embeddings = embedding_service.embed_documents(texts)

        # Add to vector store
        vector_store.add_documents(
            documents=documents,
            embeddings=embeddings,
            collection=request.collection,
        )

        # Persist
        vector_store.persist()

        return IngestResponse(
            status=IngestStatus.SUCCESS,
            documents_processed=len(documents),
            files_processed=1,
            collection=request.collection,
        )

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {request.path}")
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
    generates embeddings, and stores in the specified collection.

    Args:
        request: Directory ingest request with path and options.

    Returns:
        Ingest response with processing results.

    Raises:
        HTTPException: If directory not found or processing fails.
    """
    settings = get_settings()

    try:
        # Get services
        chunker = get_chunker(settings)
        embedding_service = get_embedding_service(settings)
        vector_store = get_vector_store(settings)

        # Process the directory
        dir_path = Path(request.path)
        if not dir_path.exists():
            raise HTTPException(status_code=404, detail=f"Directory not found: {request.path}")

        documents = chunker.process_directory(dir_path, recursive=request.recursive)

        if not documents:
            return IngestResponse(
                status=IngestStatus.SUCCESS,
                documents_processed=0,
                files_processed=0,
                collection=request.collection,
                errors=["No documents found in directory"],
            )

        # Count unique files
        files_processed = len({doc.metadata.get("source") for doc in documents})

        # Generate embeddings in batches
        texts = [doc.text for doc in documents]
        embeddings = embedding_service.embed_documents(texts)

        # Add to vector store
        vector_store.add_documents(
            documents=documents,
            embeddings=embeddings,
            collection=request.collection,
        )

        # Persist
        vector_store.persist()

        return IngestResponse(
            status=IngestStatus.SUCCESS,
            documents_processed=len(documents),
            files_processed=files_processed,
            collection=request.collection,
        )

    except DocumentProcessingError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")


@router.get("/collections")
async def list_collections() -> dict[str, Any]:
    """List all available collections.

    Returns:
        Dictionary with collection names and counts.
    """
    settings = get_settings()

    try:
        vector_store = get_vector_store(settings)
        collections = vector_store.list_collections()

        collection_info = []
        for name in collections:
            try:
                info = vector_store.get_collection_info(name)
                collection_info.append(info)
            except Exception:
                collection_info.append({"name": name, "error": "Could not get info"})

        return {
            "collections": collection_info,
            "count": len(collections),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {e}")


@router.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str) -> dict[str, str]:
    """Delete a collection and all its documents.

    Args:
        collection_name: Name of the collection to delete.

    Returns:
        Success message.

    Raises:
        HTTPException: If collection not found.
    """
    settings = get_settings()

    try:
        vector_store = get_vector_store(settings)
        vector_store.delete_collection(collection_name)
        vector_store.persist()

        return {"message": f"Collection '{collection_name}' deleted successfully"}

    except CollectionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {e}")

