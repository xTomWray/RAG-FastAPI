"""Ingest-related Pydantic schemas."""

from enum import Enum

from pydantic import BaseModel, Field


class IngestStatus(str, Enum):
    """Status of an ingest operation."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class FileIngestRequest(BaseModel):
    """Request schema for single file ingestion."""

    path: str = Field(
        ...,
        description="Path to the file to ingest",
    )
    collection: str = Field(
        default="documents",
        description="Collection to add documents to",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "path": "./data/documents/protocol.pdf",
                    "collection": "documents",
                }
            ]
        }
    }


class DirectoryIngestRequest(BaseModel):
    """Request schema for directory ingestion."""

    path: str = Field(
        ...,
        description="Path to the directory to ingest",
    )
    collection: str = Field(
        default="documents",
        description="Collection to add documents to",
    )
    recursive: bool = Field(
        default=True,
        description="Whether to process subdirectories",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "path": "./data/documents",
                    "collection": "documents",
                    "recursive": True,
                }
            ]
        }
    }


class IngestResponse(BaseModel):
    """Response schema for ingest operations."""

    status: IngestStatus = Field(description="Status of the ingest operation")
    documents_processed: int = Field(description="Number of document chunks indexed")
    files_processed: int = Field(description="Number of files processed")
    collection: str = Field(description="Collection documents were added to")
    errors: list[str] = Field(
        default_factory=list,
        description="List of errors encountered during processing",
    )
