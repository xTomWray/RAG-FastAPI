"""Query endpoint for document retrieval with hybrid vector/graph search."""

import logging
import re
import time
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from rag_service.api.v1.schemas import QueryRequest, QueryResponse, SearchResultSchema
from rag_service.config import get_settings
from rag_service.core.exceptions import CollectionNotFoundError
from rag_service.core.stats import get_stats_collector
from rag_service.dependencies import (
    get_embedding_service,
    get_graph_store,
    get_query_router,
    get_reranker,
    get_vector_store,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


class HybridQueryRequest(BaseModel):
    """Request schema for hybrid queries."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The question to search for",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of vector results to return",
    )
    collection: str = Field(
        default="documents",
        description="Collection to search in",
    )
    strategy: Literal["auto", "vector", "graph", "hybrid"] = Field(
        default="auto",
        description="Search strategy: auto (router decides), vector, graph, or hybrid",
    )
    graph_hops: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Number of hops for graph traversal",
    )
    rerank: bool | None = Field(
        default=None,
        description="Enable cross-encoder reranking (None uses config default)",
    )


class GraphContextItem(BaseModel):
    """Schema for a graph context item."""

    entity: str = Field(description="Entity name")
    entity_type: str = Field(description="Type of entity")
    relationship: str | None = Field(default=None, description="Relationship type")
    connected_to: str | None = Field(default=None, description="Connected entity")
    path: list[str] | None = Field(default=None, description="Path in graph")


class HybridQueryResponse(BaseModel):
    """Response schema for hybrid queries."""

    chunks: list[SearchResultSchema] = Field(
        default_factory=list, description="Vector search results"
    )
    graph_context: list[GraphContextItem] = Field(
        default_factory=list, description="Graph traversal results"
    )
    sources: list[str] = Field(default_factory=list, description="Unique source files")
    token_estimate: int = Field(description="Estimated token count")
    query: str = Field(description="The original query")
    collection: str = Field(description="Collection searched")
    strategy_used: str = Field(description="Strategy that was used")
    router_reasoning: str | None = Field(default=None, description="Why this strategy was chosen")


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    Uses a simple heuristic: ~4 characters per token for English text.

    Args:
        text: Text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    return len(text) // 4


def _extract_entity_mentions(query: str, graph_store: Any, collection: str) -> list[str]:
    """Extract potential entity names mentioned in the query.

    Args:
        query: User query.
        graph_store: Graph store to check entities against.
        collection: Collection to search in.

    Returns:
        List of entity names found in query.
    """
    # Get known entities from graph
    try:
        # Look for uppercase words that might be entities
        potential_entities = re.findall(r"\b([A-Z][A-Z0-9_]+)\b", query)

        # Also check for common entity types
        for entity_type in ["State", "Message", "Command", "Protocol"]:
            try:
                known = graph_store.query_by_type(entity_type, collection, limit=100)
                for entity in known:
                    if entity.name.lower() in query.lower():
                        potential_entities.append(entity.name)
            except Exception:
                continue

        return list(set(potential_entities))
    except Exception:
        return []


def _perform_vector_search(
    question: str,
    top_k: int,
    collection: str,
    enable_reranking: bool | None = None,
) -> list[SearchResultSchema]:
    """Perform vector similarity search with optional reranking.

    Args:
        question: Query text.
        top_k: Number of results.
        collection: Collection to search.
        enable_reranking: Override reranking setting (None uses config default).

    Returns:
        List of search results.

    Raises:
        CollectionNotFoundError: If collection doesn't exist.
    """
    settings = get_settings()
    embedding_service = get_embedding_service()
    vector_store = get_vector_store()

    # Validate collection exists
    collections = vector_store.list_collections()
    if collection not in collections:
        raise CollectionNotFoundError(f"Collection '{collection}' not found")

    # Determine if we should rerank
    should_rerank = enable_reranking if enable_reranking is not None else settings.enable_reranking
    reranker = get_reranker() if should_rerank else None

    # If reranking, fetch more initial results then narrow down
    initial_top_k = settings.reranker_top_k if reranker else top_k

    query_embedding = embedding_service.embed_query(question)
    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=initial_top_k,
        collection=collection,
    )

    # Apply reranking if enabled
    if reranker and results:
        logger.debug(f"Reranking {len(results)} results with cross-encoder")
        documents = [r.text for r in results]
        reranked = reranker.rerank(question, documents, top_k=top_k)

        # Rebuild results in reranked order with new scores
        reranked_results = []
        for orig_idx, score in reranked:
            r = results[orig_idx]
            reranked_results.append(
                SearchResultSchema(
                    text=r.text,
                    metadata=r.metadata,
                    score=score,  # Use cross-encoder score
                    document_id=r.document_id,
                )
            )
        return reranked_results

    return [
        SearchResultSchema(
            text=r.text,
            metadata=r.metadata,
            score=r.score,
            document_id=r.document_id,
        )
        for r in results
    ]


def _perform_graph_search(
    question: str,
    collection: str,
    hops: int,
) -> list[GraphContextItem]:
    """Perform graph traversal search.

    Args:
        question: Query text.
        collection: Collection to search.
        hops: Number of hops to traverse.

    Returns:
        List of graph context items.
    """
    settings = get_settings()
    if not settings.enable_graph_rag:
        return []

    try:
        graph_store = get_graph_store()

        # Find entities mentioned in query
        entity_mentions = _extract_entity_mentions(question, graph_store, collection)

        if not entity_mentions:
            return []

        context_items = []

        for entity_name in entity_mentions[:3]:  # Limit to 3 starting entities
            try:
                # Query neighbors
                result = graph_store.query_neighbors(
                    entity_name=entity_name,
                    hops=hops,
                    collection=collection,
                )

                # Add found entities to context
                for entity in result.entities:
                    context_items.append(
                        GraphContextItem(
                            entity=entity.name,
                            entity_type=entity.entity_type,
                        )
                    )

                # Add paths
                for path in result.paths:
                    if len(path) >= 2:
                        context_items.append(
                            GraphContextItem(
                                entity=path[0],
                                entity_type="Path",
                                connected_to=path[-1],
                                path=path,
                            )
                        )

                # Get relationships from this entity
                relationships = graph_store.query_relationships_from(  # type: ignore[union-attr]
                    entity_name=entity_name,
                    collection=collection,
                )
                for rel in relationships:
                    context_items.append(
                        GraphContextItem(
                            entity=rel["source"],
                            entity_type="Relationship",
                            relationship=rel["relationship_type"],
                            connected_to=rel["target"],
                        )
                    )

            except Exception as e:
                logger.debug(f"Graph search error for {entity_name}: {e}")
                continue

        # Deduplicate
        seen = set()
        unique_items = []
        for item in context_items:
            key = (item.entity, item.entity_type, item.connected_to)
            if key not in seen:
                seen.add(key)
                unique_items.append(item)

        return unique_items[:20]  # Limit total context items

    except Exception as e:
        logger.warning(f"Graph search failed: {e}")
        return []


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """Search for documents similar to the query (vector-only).

    This endpoint embeds the query and performs similarity search
    against the specified collection.

    Args:
        request: Query request with question and parameters.

    Returns:
        Query response with matching document chunks.

    Raises:
        HTTPException: If collection not found or query fails.
    """
    try:
        chunks = _perform_vector_search(
            question=request.question,
            top_k=request.top_k,
            collection=request.collection,
            enable_reranking=request.rerank,
        )

        # Get unique sources
        sources = list({c.metadata.get("source", "unknown") for c in chunks if c.metadata})

        # Estimate total tokens
        total_text = " ".join(c.text for c in chunks)
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


@router.post("/query/hybrid", response_model=HybridQueryResponse)
async def hybrid_query(request: HybridQueryRequest) -> HybridQueryResponse:
    """Search using hybrid vector and graph retrieval.

    This endpoint uses a smart router to determine the best strategy,
    or follows the explicitly requested strategy.

    Strategies:
    - auto: Router decides based on query analysis
    - vector: Semantic similarity search only
    - graph: Knowledge graph traversal only
    - hybrid: Both vector and graph search combined

    Args:
        request: Hybrid query request with question and strategy.

    Returns:
        Hybrid response with vector chunks and graph context.
    """
    settings = get_settings()
    start_time = time.perf_counter()
    success = True
    strategy = "unknown"
    results_count = 0

    try:
        # Determine strategy
        if request.strategy == "auto":
            query_router = get_query_router()
            explanation = query_router.get_strategy_explanation(request.question)
            strategy = str(explanation["strategy"])
            reasoning = str(explanation["reasoning"])
        else:
            strategy = request.strategy
            reasoning = f"Explicitly requested: {request.strategy}"

        chunks = []
        graph_context = []

        # Execute based on strategy
        if strategy in ("vector", "hybrid"):
            chunks = _perform_vector_search(
                question=request.question,
                top_k=request.top_k,
                collection=request.collection,
                enable_reranking=request.rerank,
            )

        if strategy in ("graph", "hybrid") and settings.enable_graph_rag:
            graph_context = _perform_graph_search(
                question=request.question,
                collection=request.collection,
                hops=request.graph_hops,
            )

        # Track results count for stats
        results_count = len(chunks) + len(graph_context)

        # Collect sources
        sources = list({c.metadata.get("source", "unknown") for c in chunks if c.metadata})

        # Estimate tokens from both sources
        vector_text = " ".join(c.text for c in chunks)
        graph_text = " ".join(
            f"{g.entity} {g.relationship or ''} {g.connected_to or ''}" for g in graph_context
        )
        token_estimate = estimate_tokens(vector_text + " " + graph_text)

        return HybridQueryResponse(
            chunks=chunks,
            graph_context=graph_context,
            sources=sources,
            token_estimate=token_estimate,
            query=request.question,
            collection=request.collection,
            strategy_used=strategy,
            router_reasoning=reasoning,
        )

    except CollectionNotFoundError as e:
        success = False
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        success = False
        logger.exception("Hybrid query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
    finally:
        # Record search stats
        duration_ms = (time.perf_counter() - start_time) * 1000
        stats = get_stats_collector()
        stats.record_search(
            duration_ms=duration_ms,
            results_count=results_count,
            strategy=strategy,
            success=success,
        )


@router.get("/query/explain")
async def explain_query_routing(question: str) -> dict[str, Any]:
    """Explain how a query would be routed.

    Args:
        question: The query to analyze.

    Returns:
        Explanation of routing decision with matched patterns.
    """
    settings = get_settings()

    try:
        query_router = get_query_router()
        explanation = query_router.get_strategy_explanation(question)

        return {
            "question": question,
            "recommended_strategy": explanation["strategy"],
            "reasoning": explanation["reasoning"],
            "matched_patterns": {
                "relational": explanation["matched_relational_patterns"],
                "semantic": explanation["matched_semantic_patterns"],
                "hybrid": explanation["matched_hybrid_patterns"],
            },
            "graph_rag_enabled": settings.enable_graph_rag,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
