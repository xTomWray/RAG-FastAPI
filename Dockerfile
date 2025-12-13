# RAG Documentation Service - Multi-stage Docker Build
# Supports CPU and GPU (CUDA) variants

# =============================================================================
# Base stage with system dependencies
# =============================================================================
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Builder stage for Python dependencies
# =============================================================================
FROM base AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first for better caching
COPY pyproject.toml ./
COPY requirements.txt ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# =============================================================================
# Production stage
# =============================================================================
FROM base AS production

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app/data/index /app/data/chroma /app/data/documents && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser src/ src/

# Switch to non-root user
USER appuser

# Set default environment variables
ENV HOST=0.0.0.0 \
    PORT=8080 \
    DEVICE=cpu \
    FAISS_INDEX_DIR=/app/data/index \
    CHROMA_PERSIST_DIR=/app/data/chroma \
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    PYTHONPATH=/app/src \
    ENABLE_GUI=true

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/health').raise_for_status()"

# Run the application
CMD ["uvicorn", "rag_service.main:app", "--host", "0.0.0.0", "--port", "8080"]

# =============================================================================
# GPU stage (CUDA support)
# =============================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS gpu-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python

FROM gpu-base AS gpu

# Copy requirements
COPY requirements.txt ./

# Create virtual environment and install dependencies with CUDA support
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app/data/index /app/data/chroma /app/data/documents && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser src/ src/

USER appuser

ENV HOST=0.0.0.0 \
    PORT=8080 \
    DEVICE=cuda \
    FAISS_INDEX_DIR=/app/data/index \
    CHROMA_PERSIST_DIR=/app/data/chroma \
    EMBEDDING_MODEL=BAAI/bge-large-en-v1.5 \
    PYTHONPATH=/app/src \
    ENABLE_GUI=true

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/health').raise_for_status()"

CMD ["uvicorn", "rag_service.main:app", "--host", "0.0.0.0", "--port", "8080"]

