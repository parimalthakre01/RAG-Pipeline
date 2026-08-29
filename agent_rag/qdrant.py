import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


class QdrantDB:
    COLLECTION_NAME = "LEGAL_DOCS"
    def __init__(self, url: str = "http://localhost:6333"):
        self.client = QdrantClient(url=url)
        self.embedding_client = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_version="2024-12-01-preview"
        )
        self.embedding_model = os.getenv("AZURE_EMBEDDING_MODEL")

    def create_collection(self, collection_name: str = "LEGAL_DOCS", vector_size: int = 1024, distance: Distance = Distance.COSINE):
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )

    def add_documents(self, collection_name: str, text_dir: str, batch_size: int = 100):
        doc_id = 0
        total = 0

        for filename in os.listdir(text_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(text_dir, filename), "r", encoding="utf-8") as f:
                    text = f.read()

                chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                batch = []

                for chunk in chunks:
                    response = self.embedding_client.embeddings.create(
                        input=chunk,
                        model=self.embedding_model
                    )
                    embedding = response.data[0].embedding

                    batch.append(PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload={"filename": filename, "text": chunk}
                    ))
                    doc_id += 1

                    if len(batch) >= batch_size:
                        self.client.upsert(collection_name=collection_name, wait=True, points=batch)
                        total += len(batch)
                        batch = []

                if batch:
                    self.client.upsert(collection_name=collection_name, wait=True, points=batch)
                    total += len(batch)

                print(f"Embedded: {filename} ({len(chunks)} chunks)")

    def search(self, collection_name: str, query_vector: list[float], limit: int = 5) -> list:
        return self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            with_payload=True,
            limit=limit,
        ).points
