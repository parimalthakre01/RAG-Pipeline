from pydantic import BaseModel
from typing import Literal


RAG_TYPE = Literal["naive", "hyde", "step_back", "iterative", "ma_rag"]


# ---------- Query ----------

class QueryRequest(BaseModel):
    question: str
    rag_type: RAG_TYPE = "naive"


class QueryResponse(BaseModel):
    question: str
    rag_type: RAG_TYPE
    answer: str


# ---------- MA-RAG verbose ----------

class MARAGRequest(BaseModel):
    question: str


class MARAGResponse(BaseModel):
    question: str
    plan: list[str]
    intermediate_answers: list[str]
    final_answer: str


# ---------- Ingest ----------

class IngestRequest(BaseModel):
    text_dir: str


class IngestResponse(BaseModel):
    status: str
    message: str
