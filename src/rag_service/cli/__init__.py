"""CLI package for the RAG Documentation Service.

This package provides a Typer-based command-line interface for managing
the RAG service, including service lifecycle, development tools, and
deployment utilities.

Example usage:
    rag-service start --reload
    rag-service test --coverage
    rag-service install --dev
"""

from rag_service.cli.main import app

__all__ = ["app"]
