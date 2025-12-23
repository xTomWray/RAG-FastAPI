"""Pytest fixtures for RAG service tests."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rag_service.config import Settings
from rag_service.core.chunker import DocumentChunker
from rag_service.core.retriever import Document


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests.

    Uses ignore_cleanup_errors=True to handle Windows file lock issues
    with ChromaDB which may not release file handles immediately.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            text="MAVLink is a communication protocol for unmanned vehicles.",
            metadata={"source": "mavlink.md", "chunk_index": 0},
            document_id="doc1",
        ),
        Document(
            text="The protocol supports message authentication and encryption.",
            metadata={"source": "mavlink.md", "chunk_index": 1},
            document_id="doc2",
        ),
        Document(
            text="ProVerif is a tool for formal verification of security protocols.",
            metadata={"source": "proverif.md", "chunk_index": 0},
            document_id="doc3",
        ),
    ]


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Create sample embeddings for testing.

    Returns normalized 384-dimensional vectors.
    """
    rng = np.random.default_rng(42)
    embeddings = rng.random((3, 384)).astype(np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


@pytest.fixture
def chunker() -> DocumentChunker:
    """Create a document chunker for testing."""
    return DocumentChunker(chunk_size=100, chunk_overlap=20)


@pytest.fixture
def test_settings(temp_dir: Path) -> Settings:
    """Create test settings with temporary directories."""
    return Settings(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        faiss_index_dir=temp_dir / "faiss",
        chroma_persist_dir=temp_dir / "chroma",
        chunk_size=100,
        chunk_overlap=20,
    )


@pytest.fixture
def sample_text_file(temp_dir: Path) -> Path:
    """Create a sample text file for testing."""
    file_path = temp_dir / "sample.txt"
    content = """
    # Sample Document

    This is a sample document for testing the RAG service.
    It contains multiple paragraphs to test chunking.

    ## Section 1

    MAVLink is a lightweight messaging protocol for communicating
    with drones and other unmanned vehicles. It is widely used
    in the drone community.

    ## Section 2

    The protocol supports various message types including heartbeat,
    command, and status messages. Each message has a specific format
    and purpose in the communication flow.
    """
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_pdf_content() -> str:
    """Sample content that would be extracted from a PDF."""
    return """
    Protocol Security Analysis

    This document describes the security analysis of the MAVLink protocol.
    We use formal methods to verify the security properties.

    The analysis reveals potential vulnerabilities in the authentication
    mechanism that should be addressed in future versions.
    """


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI app.

    Note: This requires models to be available and may be slow.
    For unit tests, prefer mocking the dependencies.
    """
    from rag_service.main import app

    with TestClient(app) as client:
        yield client
