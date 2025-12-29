"""Model search schemas for HuggingFace Hub integration."""

from pydantic import BaseModel, Field


class ModelSearchRequest(BaseModel):
    """Request schema for HuggingFace model search."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Search query for model names (e.g., 'minilm', 'bge', 'e5')",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of results to return",
    )
    filter_sentence_transformers: bool = Field(
        default=True,
        description="Filter to embedding models (feature-extraction pipeline)",
    )
    sort: str = Field(
        default="downloads",
        description="Sort field: downloads, likes, or lastModified",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "all-MiniLM",
                    "limit": 10,
                    "filter_sentence_transformers": True,
                    "sort": "downloads",
                }
            ]
        }
    }


class ModelInfo(BaseModel):
    """Schema for a single model result from HuggingFace Hub."""

    id: str = Field(description="Model ID (organization/model-name)")
    downloads: int = Field(default=0, description="Total download count")
    likes: int = Field(default=0, description="Number of likes")
    tags: list[str] = Field(default_factory=list, description="Model tags")
    pipeline_tag: str | None = Field(default=None, description="Pipeline type (e.g., sentence-similarity)")


class ModelSearchResponse(BaseModel):
    """Response schema for model search."""

    models: list[ModelInfo] = Field(description="List of matching models")
    total: int = Field(description="Total number of results returned")
    query: str = Field(description="Original search query")
    cached: bool = Field(default=False, description="Whether result was served from cache")
