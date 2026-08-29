import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "agent_rag"))
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from ma_rag.graph import build_graph


class MARAGRunner:
    def __init__(self):
        self.graph = build_graph()

    def run(self, question: str) -> str:
        result = self.graph.invoke(self._initial_state(question))
        return result["final_answer"]

    def run_verbose(self, question: str) -> dict:
        return self.graph.invoke(self._initial_state(question))

    def _initial_state(self, question: str) -> dict:
        return {
            "question":             question,
            "plan":                 [],
            "current_step":         0,
            "subquery":             "",
            "retrieved_docs":       [],
            "filtered_evidence":    "",
            "intermediate_answers": [],
            "final_answer":         "",
        }


if __name__ == "__main__":
    runner = MARAGRunner()
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter your question: ")

    print("\n--- Running MA-RAG ---")
    result = runner.run_verbose(question)

    print("\n[Plan]")
    for i, step in enumerate(result["plan"]):
        print(f"  {i+1}. {step}")

    print("\n[Intermediate Answers]")
    for i, ans in enumerate(result["intermediate_answers"]):
        print(f"  Step {i+1}: {ans}")

    print(f"\n[Final Answer]\n{result['final_answer']}")
