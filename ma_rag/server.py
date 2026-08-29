import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "agent_rag"))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from ma_rag.root import root_router

app = FastAPI(
    title="MA-RAG API",
    description="Multi-Agent RAG pipeline with LangGraph",
    default_response_class=ORJSONResponse,
)

app.include_router(root_router)


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
