from fastapi import APIRouter
from fastapi.responses import ORJSONResponse
from ma_rag.routes.query import query_router
from ma_rag.routes.ingest import ingest_router

root_router = APIRouter(
    prefix="/api/v1",
    default_response_class=ORJSONResponse,
)

root_router.include_router(query_router)
root_router.include_router(ingest_router)
