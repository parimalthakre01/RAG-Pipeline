import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv(dotenv_path=Path(__file__).parent.parent / "agent_rag" / ".env")


class AzureOpenaiClient:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
            api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
        )

    def get_client(self) -> AzureOpenAI:
        return self.client


class AzureOpenaiEmbeddingClient:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
        )

    def get_client(self) -> AzureOpenAI:
        return self.client
