from fastapi import APIRouter, HTTPException
from ma_rag.schemas import IngestRequest, IngestResponse
from ma_rag.libs.ingest import ingest_documents

ingest_router = APIRouter(prefix="/ingest", tags=["Ingest"])


@ingest_router.post("/", response_model=IngestResponse, operation_id="ingestDocuments")
def ingest(request: IngestRequest):
    try:
        result = ingest_documents(request.text_dir)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return IngestResponse(
        status="success",
        message=f"Ingested {result['count']} file(s) from {result['dir']}",
    )
