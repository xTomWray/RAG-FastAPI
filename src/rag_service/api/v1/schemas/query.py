"""Query-related Pydantic schemas."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request schema for document queries."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The question to search for relevant documents",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of results to return",
    )
    collection: str = Field(
        default="documents",
        description="Collection to search in",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "How does MAVLink authentication work?",
                    "top_k": 5,
                    "collection": "documents",
                }
            ]
        }
    }


class SearchResultSchema(BaseModel):
    """Schema for a single search result."""

    text: str = Field(description="The document chunk text")
    metadata: dict = Field(description="Document metadata")
    score: float = Field(description="Similarity score (0-1)")
    document_id: str = Field(description="Unique document identifier")


class QueryResponse(BaseModel):
    """Response schema for document queries."""

    chunks: list[SearchResultSchema] = Field(description="Retrieved document chunks")
    sources: list[str] = Field(description="Unique source files")
    token_estimate: int = Field(
        description="Estimated token count for the returned context"
    )
    query: str = Field(description="The original query")
    collection: str = Field(description="Collection searched")

