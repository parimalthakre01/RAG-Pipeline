import os
from ma_rag.state import MARAGState
from ma_rag.azure_client import AzureOpenaiClient
from ma_rag.prompts import STEP_DEFINER_SYSTEM, STEP_DEFINER_USER

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = AzureOpenaiClient().get_client()
    return _llm


def step_definer_node(state: MARAGState) -> dict:
    llm = _get_llm()
    current_subtask = state["plan"][state["current_step"]]
    prior_answers = state.get("intermediate_answers", [])
    prior_text = (
        "\n".join(f"Step {i+1}: {ans}" for i, ans in enumerate(prior_answers))
        if prior_answers
        else "None yet."
    )

    resp = llm.chat.completions.create(
        model=os.getenv("AZURE_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": STEP_DEFINER_SYSTEM},
            {"role": "user",   "content": STEP_DEFINER_USER.format(
                question=state["question"],
                current_subtask=current_subtask,
                prior_answers=prior_text,
            )},
        ],
        temperature=0.0,
    )

    return {"subquery": resp.choices[0].message.content.strip()}
