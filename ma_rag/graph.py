from langgraph.graph import StateGraph, END, START

from ma_rag.state import MARAGState
from ma_rag.retrieval import RetrievalTool
from ma_rag.agents.planner import planner_node
from ma_rag.agents.step_definer import step_definer_node
from ma_rag.agents.extractor import extractor_node
from ma_rag.agents.qa_agent import qa_agent_node
from ma_rag.agents.synthesizer import synthesizer_node

_retrieval_tool = RetrievalTool()


def retrieval_node(state: MARAGState) -> dict:
    docs = _retrieval_tool.search(state["subquery"], top_k=5)
    return {"retrieved_docs": docs}


def route_after_qa(state: MARAGState) -> str:
    if state["current_step"] < len(state["plan"]):
        return "step_definer"
    return "synthesizer"


def build_graph():
    g = StateGraph(MARAGState)

    g.add_node("planner",      planner_node)
    g.add_node("step_definer", step_definer_node)
    g.add_node("retrieval",    retrieval_node)
    g.add_node("extractor",    extractor_node)
    g.add_node("qa_agent",     qa_agent_node)
    g.add_node("synthesizer",  synthesizer_node)

    g.add_edge(START,          "planner")
    g.add_edge("planner",      "step_definer")
    g.add_edge("step_definer", "retrieval")
    g.add_edge("retrieval",    "extractor")
    g.add_edge("extractor",    "qa_agent")
    g.add_edge("synthesizer",  END)

    g.add_conditional_edges(
        "qa_agent",
        route_after_qa,
        {
            "step_definer": "step_definer",
            "synthesizer":  "synthesizer",
        },
    )

    return g.compile()
