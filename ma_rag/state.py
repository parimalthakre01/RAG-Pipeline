import operator
from typing import Annotated
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
