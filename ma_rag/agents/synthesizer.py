import os
from ma_rag.state import MARAGState
from ma_rag.azure_client import AzureOpenaiClient
from ma_rag.prompts import SYNTHESIZER_SYSTEM, SYNTHESIZER_USER

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = AzureOpenaiClient().get_client()
    return _llm


def synthesizer_node(state: MARAGState) -> dict:
    llm = _get_llm()
    answers_text = "\n".join(
        f"[Step {i+1}] {ans}" for i, ans in enumerate(state["intermediate_answers"])
    )

    resp = llm.chat.completions.create(
        model=os.getenv("AZURE_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": SYNTHESIZER_SYSTEM},
            {"role": "user",   "content": SYNTHESIZER_USER.format(
                question=state["question"],
                intermediate_answers=answers_text,
            )},
        ],
        temperature=0.0,
    )

    return {"final_answer": resp.choices[0].message.content.strip()}
