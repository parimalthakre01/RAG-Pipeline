import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "agent_rag"))

from agent_rag import ask_query
from ma_rag.main import MARAGRunner

_agent = None
_ma_rag_runner = None


def _get_agent() -> ask_query:
    global _agent
    if _agent is None:
        _agent = ask_query()
    return _agent


def _get_ma_rag_runner() -> MARAGRunner:
    global _ma_rag_runner
    if _ma_rag_runner is None:
        _ma_rag_runner = MARAGRunner()
    return _ma_rag_runner


def run_query(question: str, rag_type: str) -> str:
    if rag_type == "ma_rag":
        return _get_ma_rag_runner().run(question)
    return _get_agent().get_response(question, rag_type=rag_type)


def run_ma_rag_verbose(question: str) -> dict:
    return _get_ma_rag_runner().run_verbose(question)
