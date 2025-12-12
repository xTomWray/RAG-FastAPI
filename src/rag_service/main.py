"""FastAPI application factory and configuration."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_service.api.v1.router import router as v1_router
from rag_service.config import get_settings


def setup_logging() -> None:
    """Configure logging for the application."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    setup_logging()
    logger = logging.getLogger(__name__)
    settings = get_settings()

    logger.info("Starting RAG Documentation Service")
    logger.info(f"Embedding model: {settings.embedding_model}")
    logger.info(f"Device: {settings.get_resolved_device()}")
    logger.info(f"Vector store: {settings.vector_store_backend}")

    # Pre-load models on startup (optional, can be lazy loaded)
    try:
        from rag_service.dependencies import get_embedding_service, get_vector_store

        logger.info("Loading embedding model...")
        _ = get_embedding_service(settings)
        logger.info("Embedding model loaded successfully")

        logger.info("Initializing vector store...")
        _ = get_vector_store(settings)
        logger.info("Vector store initialized")

    except Exception as e:
        logger.warning(f"Failed to pre-load models: {e}")
        logger.warning("Models will be loaded on first request")

    yield

    # Shutdown
    logger.info("Shutting down RAG Documentation Service")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="RAG Documentation Service",
        description=(
            "A cross-platform RAG API service for document retrieval and question answering. "
            "Supports PDF, Markdown, XML, code files, and more."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API router with prefix
    app.include_router(v1_router, prefix=settings.api_prefix)

    # Also include health endpoints at root level for k8s probes
    from rag_service.api.v1.endpoints import health

    app.include_router(health.router)

    return app


# Create the app instance
app = create_app()

