import os
from pathlib import Path
from dotenv import load_dotenv
from azure_setup import AzureOpenaiEmbeddingClient, AzureOpenaiClient
from qdrant import QdrantDB


load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

class BaseRAG: 
    SYSTEM = (
        "You are a legal assistant. Answer using the provided context only. "
        "Always cite the source document."
    )
    
    def __init__(self):
        self.llm = AzureOpenaiClient().get_client()
        self.embedding_client = AzureOpenaiEmbeddingClient().get_client()
        self.db = QdrantDB()
        self.embedding_model = os.getenv("AZURE_EMBEDDING_MODEL")
        self.llm_model = os.getenv("AZURE_DEPLOYMENT")
        
    def _embed(self, text:str)-> list[float]:
        response = self.embedding_client.embeddings.create(
            input = text,
            model = self.embedding_model
        )
        return response.data[0].embedding
    
    def _retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        vector = self._embed(query)
        points = self.db.search(QdrantDB.COLLECTION_NAME, vector, limit=top_k)
        return [
            {
                "text": p.payload["text"],
                "source": p.payload["filename"],
                "score": round(p.score, 4),
            }
            for p in points
        ]
    
    def _format_context(self, docs: list[dict]) -> str:
        return "\n\n---\n\n".join(
            f"[Source: {d['source']} | score: {d['score']}]\n{d['text']}"
            for d in docs
        )
    
    def _chat(self, messages: list[dict], tools: list | None = None, tool_choice=None):
        kwargs = dict(model=self.llm_model, messages=messages)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        return self.llm.chat.completions.create(**kwargs)

    def run(self, question: str) -> str:
        raise NotImplementedError