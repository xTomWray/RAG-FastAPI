"""Custom exceptions for the RAG service."""


class RAGServiceError(Exception):
    """Base exception for RAG service errors."""

    pass


class DocumentNotFoundError(RAGServiceError):
    """Raised when a document is not found."""

    pass


class CollectionNotFoundError(RAGServiceError):
    """Raised when a collection is not found."""

    pass


class CollectionExistsError(RAGServiceError):
    """Raised when trying to create a collection that already exists."""

    pass


class EmbeddingError(RAGServiceError):
    """Raised when embedding generation fails."""

    pass


class IndexError(RAGServiceError):
    """Raised when vector index operations fail."""

    pass


class DocumentProcessingError(RAGServiceError):
    """Raised when document processing fails."""

    pass


class UnsupportedFileTypeError(DocumentProcessingError):
    """Raised when attempting to process an unsupported file type."""

    pass

