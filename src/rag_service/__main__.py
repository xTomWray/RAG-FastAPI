"""Entry point for running the RAG service as a module."""

import uvicorn

from rag_service.config import get_settings


def main() -> None:
    """Run the RAG service."""
    settings = get_settings()
    uvicorn.run(
        "rag_service.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()

