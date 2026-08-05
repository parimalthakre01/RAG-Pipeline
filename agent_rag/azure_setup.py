import os
from openai import AzureOpenAI

class AzureOpenaiClient: 
    def __init__(self):
        self.client = AzureOpenAI(
            api_version="2024-12-01-preview",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_DEPLOYMENT")
        )

    def get_client(self) -> AzureOpenAI:
        return self.client

class AzureOpenaiEmbeddingClient:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_version="2024-12-01-preview"
        )
    def get_client(self) -> AzureOpenAI:
        return self.client

