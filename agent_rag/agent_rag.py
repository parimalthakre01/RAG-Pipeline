import os
from dotenv import load_dotenv
from azure_setup import AzureOpenaiEmbeddingClient, AzureOpenaiClient
from qdrant import QdrantDB
from rag.hyde import HydeRag
from rag.iterative import interativeRag
from rag.step_back import StepBack
from rag.naive import NaiveRAG
load_dotenv()

RAG_MAP = {
    "naive":NaiveRAG, 
    "hyde":HydeRag,
    "step_back": StepBack,
    "iterative": interativeRag,
}

class ask_query(): 
    def __init__(self):
        self.llm = AzureOpenaiClient().get_client()
        self.embedding_client = AzureOpenaiEmbeddingClient().get_client()
        self.db = QdrantDB()
        self.embedding_model = os.getenv("AZURE_EMBEDDING_MODEL")
        self.llm_model = os.getenv("AZURE_DEPLOYMENT")
    
    def check_collection(self, collection_name: str, vector_size: int):
        collection = self.db.create_collection(collection_name, vector_size)
        if collection:
            print("collection created successfully")
        
    def get_embedding(self, text: str): 
        response = self.embedding_client.embeddings.create(
            input = text, 
            model = self.embedding_model
        )
        
        return response.data[0].embedding
    
    def get_response(self, user_query: str, rag_type: str = "naive"): 
        return RAG_MAP[rag_type]().run(user_query)
    
if __name__ == "__main__":
    agent = ask_query()
    agent.check_collection("LEGAL_DOCS", 1536)
    agent.db.add_documents("LEGAL_DOCS", os.path.join("..", "text_data"))
    user_query = input("Enter your query: ")
    answer = agent.get_response(user_query)
    print(answer)