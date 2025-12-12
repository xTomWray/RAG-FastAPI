"""Query endpoint for document retrieval."""

from fastapi import APIRouter, HTTPException

from rag_service.api.v1.schemas import QueryRequest, QueryResponse, SearchResultSchema
from rag_service.config import get_settings
from rag_service.core.exceptions import CollectionNotFoundError
from rag_service.dependencies import get_embedding_service, get_vector_store

router = APIRouter(tags=["query"])


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    Uses a simple heuristic: ~4 characters per token for English text.

    Args:
        text: Text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    return len(text) // 4


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """Search for documents similar to the query.

    This endpoint embeds the query and performs similarity search
    against the specified collection.

    Args:
        request: Query request with question and parameters.

    Returns:
        Query response with matching document chunks.

    Raises:
        HTTPException: If collection not found or query fails.
    """
    settings = get_settings()

    try:
        # Get services
        embedding_service = get_embedding_service(settings)
        vector_store = get_vector_store(settings)

        # Embed the query
        query_embedding = embedding_service.embed_query(request.question)

        # Search for similar documents
        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            collection=request.collection,
        )

        # Convert to response schema
        chunks = [
            SearchResultSchema(
                text=r.text,
                metadata=r.metadata,
                score=r.score,
                document_id=r.document_id,
            )
            for r in results
        ]

        # Get unique sources
        sources = list(
            {r.metadata.get("source", "unknown") for r in results if r.metadata}
        )

        # Estimate total tokens
        total_text = " ".join(r.text for r in results)
        token_estimate = estimate_tokens(total_text)

        return QueryResponse(
            chunks=chunks,
            sources=sources,
            token_estimate=token_estimate,
            query=request.question,
            collection=request.collection,
        )

    except CollectionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

