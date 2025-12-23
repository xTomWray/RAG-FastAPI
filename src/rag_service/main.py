"""FastAPI application factory and configuration."""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from rag_service.api.v1.router import router as v1_router
from rag_service.config import get_settings
from rag_service.core.logging import configure_logging, get_logger
from rag_service.middleware.correlation import CorrelationIdMiddleware

# Check if GUI should be enabled (default: True)
ENABLE_GUI = os.environ.get("ENABLE_GUI", "true").lower() not in ("false", "0", "no")


def setup_logging() -> None:
    """Configure structured logging for the application."""
    settings = get_settings()
    # Use console format in development (LOG_FORMAT=console), JSON in production
    log_format = os.environ.get("LOG_FORMAT", "json")
    if log_format not in ("json", "console"):
        log_format = "json"
    configure_logging(
        log_format=log_format,  # type: ignore[arg-type]
        log_level=settings.log_level,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    setup_logging()
    logger = get_logger(__name__)
    settings = get_settings()

    logger.info(
        "Starting RAG Documentation Service",
        embedding_model=settings.embedding_model,
        device=settings.get_resolved_device(),
        vector_store=settings.vector_store_backend,
    )

    # Pre-load models on startup (optional, can be lazy loaded)
    try:
        from rag_service.dependencies import get_embedding_service, get_vector_store

        logger.info("Loading embedding model")
        _ = get_embedding_service()
        logger.info("Embedding model loaded successfully")

        logger.info("Initializing vector store")
        _ = get_vector_store()
        logger.info("Vector store initialized")

    except Exception as e:
        logger.warning("Failed to pre-load models", error=str(e))
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
    logger = get_logger(__name__)

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

    # Add correlation ID middleware for request tracing
    app.add_middleware(CorrelationIdMiddleware)

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

    # Serve manifest.json to prevent 404 errors from browser PWA detection
    @app.get("/manifest.json", include_in_schema=False)
    async def manifest():
        return JSONResponse({
            "name": "RAG Documentation Service",
            "short_name": "RAG Service",
            "description": "Document retrieval and question answering",
            "start_url": "/ui",
            "display": "standalone",
            "background_color": "#ffffff",
            "theme_color": "#1976d2",
        })

    # Mount Gradio UI if enabled
    if ENABLE_GUI:
        try:
            import gradio as gr

            from rag_service.ui.app import create_ui

            # Create Gradio interface pointing to localhost API
            gradio_app = create_ui(api_url=f"http://localhost:{settings.port}")

            # Mount at /ui path
            app = gr.mount_gradio_app(app, gradio_app, path="/ui")

            # Redirect root to UI
            @app.get("/", include_in_schema=False)
            async def redirect_to_ui():
                return RedirectResponse(url="/ui")

            logger.info("Gradio UI mounted at /ui")

        except ImportError:
            logger.warning("Gradio not installed, UI disabled")
        except Exception as e:
            logger.warning(f"Failed to mount Gradio UI: {e}")

    return app


# Create the app instance
app = create_app()

