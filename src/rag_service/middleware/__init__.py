"""Middleware package for the RAG service.

Provides FastAPI middleware components for request processing,
including correlation IDs for distributed tracing.
"""

from rag_service.middleware.correlation import (
    CorrelationIdMiddleware,
    get_correlation_id,
)

__all__ = [
    "CorrelationIdMiddleware",
    "get_correlation_id",
]
