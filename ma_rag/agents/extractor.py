import os
from ma_rag.state import MARAGState
from ma_rag.azure_client import AzureOpenaiClient
from ma_rag.prompts import EXTRACTOR_SYSTEM, EXTRACTOR_USER

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = AzureOpenaiClient().get_client()
    return _llm


def _format_passages(docs: list) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {d['source']} | score: {d['score']}]\n{d['text']}"
        for d in docs
    )


def extractor_node(state: MARAGState) -> dict:
    llm = _get_llm()
    current_subtask = state["plan"][state["current_step"]]
    passages = _format_passages(state["retrieved_docs"])

    resp = llm.chat.completions.create(
        model=os.getenv("AZURE_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": EXTRACTOR_SYSTEM},
            {"role": "user",   "content": EXTRACTOR_USER.format(
                current_subtask=current_subtask,
                passages=passages,
            )},
        ],
        temperature=0.0,
    )

    return {"filtered_evidence": resp.choices[0].message.content.strip()}
