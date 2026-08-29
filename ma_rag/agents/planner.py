import json
import os
from ma_rag.state import MARAGState
from ma_rag.azure_client import AzureOpenaiClient
from ma_rag.prompts import PLANNER_SYSTEM, PLANNER_USER

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = AzureOpenaiClient().get_client()
    return _llm


def planner_node(state: MARAGState) -> dict:
    llm = _get_llm()
    resp = llm.chat.completions.create(
        model=os.getenv("AZURE_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user",   "content": PLANNER_USER.format(question=state["question"])},
        ],
        temperature=0.0,
    )

    raw = resp.choices[0].message.content.strip()

    # Strip markdown fences if the model adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        plan = json.loads(raw)
        if not isinstance(plan, list) or not all(isinstance(s, str) for s in plan):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        plan = [state["question"]]

    return {"plan": plan, "current_step": 0}
