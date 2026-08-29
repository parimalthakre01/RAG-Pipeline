# MA-RAG Implementation Plan (LangGraph)

## Context

Building MA-RAG from the paper "MA-RAG: Multi-Agent Retrieval-Augmented Generation" (2505.20096v2) as a new `ma_rag/` module alongside the existing `agent_rag/` code. MA-RAG replaces the single retrieve-then-generate step with five specialized agents coordinated by a LangGraph `StateGraph`.

**LLM:** Azure OpenAI (GPT-4o-mini) — matches existing `agent_rag/` setup  
**Vector store:** Qdrant + Azure embedding model — reuses `agent_rag/qdrant.py`  
**Installed:** `langgraph==1.2.11`, `langchain==1.3.18`, `openai` (transitive), `langchain-openai` (transitive)  
**Missing:** `qdrant-client` — run `pip install qdrant-client>=1.9.0` before starting

---

## File Structure

```
ma_rag/
├── __init__.py
├── state.py            # MARAGState TypedDict
├── azure_client.py     # Mirrors agent_rag/azure_setup.py (with env path fix)
├── retrieval.py        # RetrievalTool wrapping QdrantDB.search()
├── prompts.py          # All LLM prompt templates for the 5 agents
├── graph.py            # StateGraph: nodes, edges, conditional routing
├── agents/
│   ├── __init__.py
│   ├── planner.py
│   ├── step_definer.py
│   ├── extractor.py
│   ├── qa_agent.py
│   └── synthesizer.py
└── main.py             # MARAGRunner entry point
```

Also update: `requirements.txt` — add `qdrant-client>=1.9.0`

---

## Graph Flow

```
START → [planner] → [step_definer] → [retrieval] → [extractor] → [qa_agent]
                          ↑                                           │
                          └──────── current_step < len(plan) ─────────┤
                                                                       │
                                    current_step == len(plan) ─────────┘
                                                                       ↓
                                                              [synthesizer] → END
```

Conditional routing lives in `route_after_qa(state)` inside `graph.py`:
- `state["current_step"] < len(state["plan"])` → `"step_definer"` (loop back)
- otherwise → `"synthesizer"` (done)

`current_step` is incremented inside `qa_agent_node` so the routing function sees the updated value immediately.

---

## State (`ma_rag/state.py`)

```python
from typing import Annotated
import operator
from typing_extensions import TypedDict

class MARAGState(TypedDict):
    question:             str
    plan:                 list[str]
    current_step:         int
    subquery:             str
    retrieved_docs:       list[dict]                      # overwritten each step
    filtered_evidence:    str
    intermediate_answers: Annotated[list[str], operator.add]  # appended each step
    final_answer:         str
```

`retrieved_docs` is a plain overwrite (step-local). `intermediate_answers` uses `operator.add` so each `qa_agent_node` call appends rather than replaces.

---

## Agent Nodes

Each agent file has a module-level `_llm = None` singleton (lazy-init). Every node signature: `def xxx_node(state: MARAGState) -> dict`.

| Node | Reads | Writes |
|------|-------|--------|
| `planner_node` | `question` | `plan`, `current_step=0` |
| `step_definer_node` | `question`, `plan`, `current_step`, `intermediate_answers` | `subquery` |
| `retrieval_node` (in graph.py) | `subquery` | `retrieved_docs` |
| `extractor_node` | `plan`, `current_step`, `retrieved_docs` | `filtered_evidence` |
| `qa_agent_node` | `question`, `plan`, `current_step`, `filtered_evidence` | `intermediate_answers` (+1), `current_step` (+1) |
| `synthesizer_node` | `question`, `intermediate_answers` | `final_answer` |

### Prompts (in `prompts.py`)

- **Planner:** Returns ONLY a JSON array of subtask strings. Fallback: if JSON parse fails, use `[question]` as single-step plan.
- **Step Definer:** Produces one concrete search query conditioned on the current subtask + all prior answers.
- **Extractor:** Keeps only sentences/spans directly relevant to the current subtask. Returns `NO_RELEVANT_EVIDENCE` if nothing fits.
- **QA Agent:** One-to-two sentence answer using only filtered evidence. Returns `INSUFFICIENT_EVIDENCE: <what's missing>` if evidence is lacking.
- **Synthesizer:** Integrates all intermediate answers into a single coherent final response.

All agents use `temperature=0.0`.

---

## Azure Client (`ma_rag/azure_client.py`)

Mirror `agent_rag/azure_setup.py` with two corrections:
1. Load `.env` via explicit path: `dotenv_path=Path(__file__).parent.parent / "agent_rag" / ".env"`
2. Use `os.getenv("AZURE_OPENAI_API_KEY")` — existing `qdrant.py` has a bug using `"AZURE_API_KEY"` (wrong key name); do not copy this bug.

---

## Retrieval Tool (`ma_rag/retrieval.py`)

Add `agent_rag/` to `sys.path` so `from qdrant import QdrantDB` works. Wrap `QdrantDB.search()` with embed-then-search, returning `[{"text", "source", "score"}]` — same shape as `BaseRAG._retrieve()`.

```python
sys.path.insert(0, str(Path(__file__).parent.parent / "agent_rag"))
from qdrant import QdrantDB
```

---

## Graph Assembly (`ma_rag/graph.py`)

```python
from langgraph.graph import StateGraph, END, START

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

    g.add_conditional_edges("qa_agent", route_after_qa,
        {"step_definer": "step_definer", "synthesizer": "synthesizer"})

    return g.compile()
```

---

## Entry Point (`ma_rag/main.py`)

Class `MARAGRunner`:
- `run(question) -> str` — returns `final_answer` only
- `run_verbose(question) -> dict` — returns full state (plan + intermediate answers + final answer)

CLI: `python ma_rag/main.py "your question here"`

---

## Existing Code to Reuse

| File | What to reuse |
|------|--------------|
| `agent_rag/azure_setup.py` | Pattern for `AzureOpenaiClient` / `AzureOpenaiEmbeddingClient` |
| `agent_rag/qdrant.py` | `QdrantDB` class (`search()`, `add_documents()`) via sys.path |
| `agent_rag/rag/base.py` | `_format_context()` pattern (replicate inline, don't import) |
| `agent_rag/.env` | All credentials (loaded via explicit `dotenv_path`) |

---

## Verification

1. **Install:** `pip install qdrant-client>=1.9.0`
2. **Unit test routing:**
   ```python
   route_after_qa({"plan": ["a","b"], "current_step": 1})  # → "step_definer"
   route_after_qa({"plan": ["a","b"], "current_step": 2})  # → "synthesizer"
   ```
3. **Smoke test planner:** Call `planner_node({"question": "..."})` and verify `plan` is a non-empty list
4. **Integration test** (requires Qdrant running + `LEGAL_DOCS` collection populated):
   ```bash
   python ma_rag/main.py "What are the termination provisions in the employment contracts?"
   ```
   Inspect: `plan` (expect 1–3 steps), `intermediate_answers` (one per step), `final_answer`
5. **Compare:** Run the same question through `agent_rag/agent_rag.py` with `rag_type="naive"` and compare answer quality
