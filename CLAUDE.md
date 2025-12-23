# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG Documentation Service - A Python FastAPI-based Retrieval-Augmented Generation system for document ingestion, semantic search, and question-answering. Supports both vector and graph-based retrieval strategies with cross-platform deployment (Windows, Linux, macOS).

## Common Commands

All commands are available via the Python CLI or Makefile (thin wrapper):

```bash
# Service Lifecycle
rag-service start                    # Start with UI
rag-service start --reload           # Development mode with hot-reload
rag-service start --no-ui -w 4       # Production API-only (4 workers)
rag-service stop                     # Stop running service
rag-service status                   # Check service status

# Installation
rag-service install                  # Production dependencies
rag-service install --dev            # Development environment
rag-service install --gpu            # GPU support (CUDA)

# Testing
rag-service test                     # All tests
rag-service test --unit              # Unit tests only
rag-service test --coverage          # With coverage report

# Code Quality
rag-service check                    # All checks (lint + format + typecheck)
rag-service lint --fix               # Auto-fix linting issues
rag-service format                   # Auto-format code

# Single test execution
pytest tests/unit/test_chunker.py -v
pytest tests/unit/test_chunker.py::TestDocumentChunker::test_chunk_text -v

# Docker
rag-service docker-build             # Build image
rag-service docker-run               # Run container

# Makefile shortcuts (all delegate to Python CLI)
make dev              # Install dev dependencies
make run              # Start with hot-reload
make test             # Run all tests
```

## Architecture

### Core Flow
```
Request → FastAPI (main.py) → API Router (api/v1/)
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
            /ingest endpoints              /query endpoint
                    ↓                               ↓
            DocumentChunker              QueryRouter.classify()
            (core/chunker.py)            (core/router.py)
                    ↓                               ↓
            Embeddings Service           Route: vector/graph/hybrid
            (core/embeddings.py)                    ↓
                    ↓                       VectorStore.search()
            VectorStore.add()              and/or GraphStore
            (infrastructure/)
```

### Key Layers
- **cli/**: Typer-based CLI commands (start, stop, test, lint, etc.)
- **api/v1/endpoints/**: FastAPI route handlers (query.py, ingest.py, health.py)
- **api/v1/schemas/**: Pydantic request/response models
- **core/**: Business logic (chunker, embeddings, router, logging, async_utils)
- **infrastructure/**: Data adapters (faiss_store, chroma_store, neo4j_store, cache)
- **middleware/**: Request processing (correlation IDs for tracing)
- **ui/**: Gradio web interface (standalone or embedded)

### Dependency Injection
All services use singleton pattern via `dependencies.py`:
```python
from rag_service.dependencies import get_vector_store, get_embedding_service
# FastAPI: Depends(get_vector_store)
# Direct: get_vector_store_direct()
```

### Configuration Priority
1. Environment variables (highest)
2. `.env` file
3. `config.yaml` (persisted, auto-created)
4. Code defaults (lowest)

Device auto-detection: CUDA → MPS (Apple Silicon) → CPU

## File Type Support

DocumentChunker handles 40+ file types including PDF, DOCX, Markdown, HTML, XML, and source code. Uses `unstructured` library with graceful fallbacks.

## Vector Store Backends

- **FAISS** (default): Fast, large-scale, no filtering
- **ChromaDB**: Simpler API, metadata filtering support

## GraphRAG (Optional)

Disabled by default. When enabled:
- Entity extraction via rule-based patterns or Ollama LLM
- Graph storage via NetworkX (dev) or Neo4j (production)
- Query routing determines vector/graph/hybrid strategy

## Testing Strategy

- **Unit tests** (`tests/unit/`): Isolated component testing
- **Integration tests** (`tests/integration/`): Full API flows
- Integration tests may be flaky in CI (marked continue-on-error)

## Entry Points

- **API**: `http://localhost:8080/docs` (Swagger UI)
- **Web UI**: `http://localhost:8080/ui` (Gradio interface)
- **CLI**: `rag-service [start|stop|test|lint|...]` or `python -m rag_service`

## Deployment Patterns

```bash
# All-in-One (Development)
rag-service start --reload

# API-Only (Production Backend)
rag-service start --no-ui --workers 4

# Separate UI (Scaled Frontend)
rag-service start --no-ui --port 8080  # Terminal 1
rag-ui --api-url http://api:8080       # Terminal 2

# Docker
docker run -p 8080:8080 rag-documentation-service
```

## Production Features

- **Structured Logging**: JSON format for log aggregation (set `LOG_FORMAT=console` for dev)
- **Correlation IDs**: `X-Correlation-ID` header for distributed tracing
- **Caching**: Optional in-memory or Redis caching (`pip install .[cache]`)
- **Async Processing**: CPU-bound operations run in ThreadPoolExecutor
