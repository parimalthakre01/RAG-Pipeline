import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "agent_rag"))

from qdrant import QdrantDB
from ma_rag.azure_client import AzureOpenaiEmbeddingClient


class RetrievalTool:
    COLLECTION_NAME = QdrantDB.COLLECTION_NAME

    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self.db = QdrantDB(url=qdrant_url)
        self.embedding_client = AzureOpenaiEmbeddingClient().get_client()
        self.embedding_model = os.getenv("AZURE_EMBEDDING_MODEL")

    def _embed(self, text: str) -> list:
        resp = self.embedding_client.embeddings.create(
            input=text,
            model=self.embedding_model,
        )
        return resp.data[0].embedding

    def search(self, query: str, top_k: int = 5) -> list:
        vector = self._embed(query)
        points = self.db.search(self.COLLECTION_NAME, vector, limit=top_k)
        return [
            {
                "text": p.payload["text"],
                "source": p.payload["filename"],
                "score": round(p.score, 4),
            }
            for p in points
        ]
