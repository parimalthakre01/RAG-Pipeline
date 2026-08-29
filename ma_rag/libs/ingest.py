import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "agent_rag"))

from dotenv import load_dotenv

from qdrant import QdrantDB


def ingest_documents(text_dir: str) -> dict:
    path = Path(text_dir)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Directory not found: {text_dir}")

    txt_files = list(path.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in: {text_dir}")

    db = QdrantDB()
    db.add_documents(QdrantDB.COLLECTION_NAME, str(path))

    return {"count": len(txt_files), "dir": str(path)}
