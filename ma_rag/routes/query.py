from fastapi import APIRouter, HTTPException
from ma_rag.schemas import QueryRequest, QueryResponse, MARAGRequest, MARAGResponse
from ma_rag.libs.query import run_query, run_ma_rag_verbose

query_router = APIRouter(prefix="/query", tags=["Query"])


@query_router.post("/", response_model=QueryResponse, operation_id="runQuery")
def query(request: QueryRequest):
    try:
        answer = run_query(request.question, request.rag_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        question=request.question,
        rag_type=request.rag_type,
        answer=answer,
    )


@query_router.post("/ma-rag/verbose", response_model=MARAGResponse, operation_id="runMARAGVerbose")
def ma_rag_verbose(request: MARAGRequest):
    try:
        result = run_ma_rag_verbose(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return MARAGResponse(
        question=request.question,
        plan=result["plan"],
        intermediate_answers=result["intermediate_answers"],
        final_answer=result["final_answer"],
    )
