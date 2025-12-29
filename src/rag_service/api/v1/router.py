"""API v1 router that aggregates all endpoints."""

from fastapi import APIRouter

from rag_service.api.v1.endpoints import health, ingest, models, query

router = APIRouter()

# Include all endpoint routers
router.include_router(health.router)
router.include_router(query.router)
router.include_router(ingest.router)
router.include_router(models.router)
