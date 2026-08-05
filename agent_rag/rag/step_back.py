# User question (specific)
#         │
#         ▼
# LLM generates a broader/abstract version of the question
#         │
#         ▼
# retrieve on BOTH questions (specific + abstract)
#         │
#         ▼
# merge + deduplicate chunks
#         │
#         ▼
# LLM answers the original specific question with richer context
from rag.base import BaseRAG

class StepBack(BaseRAG):
    def run(self, question: str) -> str:
        sb_resp = self._chat([
            {
                "role": "system",
                "content": (
                    "Given a specific question, produce a broader, more general version "
                    "that captures the underlying concept or principle. "
                    "Return only the general question, nothing else."
                ),
            },
            {"role": "user", "content": f"Specific question: {question}"},
        ])
        abstract_question = sb_resp.choices[0].message.content
        
        specific_docs = self._retrieve(question, top_k=3)
        abstract_docs = self._retrieve(abstract_question, top_k=3)
        
        seen, merged = set(), []
        for doc in specific_docs + abstract_docs:
            if doc["text"] not in seen:
                seen.add(doc["text"])
                merged.append(doc)
                
        context = self._format_context(merged)
        resp = self._chat([
            {"role": "system", "content": self.SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"  
                ),
            },
        ])
        return resp.choices[0].message.content