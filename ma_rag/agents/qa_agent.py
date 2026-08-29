import os
from ma_rag.state import MARAGState
from ma_rag.azure_client import AzureOpenaiClient
from ma_rag.prompts import QA_AGENT_SYSTEM, QA_AGENT_USER

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = AzureOpenaiClient().get_client()
    return _llm


def qa_agent_node(state: MARAGState) -> dict:
    llm = _get_llm()
    current_subtask = state["plan"][state["current_step"]]

    resp = llm.chat.completions.create(
        model=os.getenv("AZURE_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": QA_AGENT_SYSTEM},
            {"role": "user",   "content": QA_AGENT_USER.format(
                question=state["question"],
                current_subtask=current_subtask,
                evidence=state["filtered_evidence"],
            )},
        ],
        temperature=0.0,
    )

    step_answer = resp.choices[0].message.content.strip()

    # intermediate_answers uses operator.add reducer — returning a list appends it
    # current_step is a plain field — returning the incremented value overwrites it
    return {
        "intermediate_answers": [step_answer],
        "current_step": state["current_step"] + 1,
    }
