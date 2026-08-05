from rag.base import BaseRAG

class HydeRag(BaseRAG):
    """
        Hypothetical Document Embeddings.

        Problem it solves: the user's question uses different vocabulary than
        the documents (e.g. question says "termination clause", docs say
        "contract dissolution provision").  Embedding the question directly
        gives poor retrieval.

        Fix: ask the LLM to write a short hypothetical answer, then embed
        *that* for retrieval.  The hypothetical answer uses the same vocabulary
        as real documents, so cosine similarity is much higher.
    """
    
    def run(self, question: str) -> str: 
        hyp_resp = self._chat(
            [
                {
                    "role" : "user", "content" : (
                        "Write a short paragraph that directly answers the following "
                        "question as if you already know the answer. "
                        "This is only used for document search, not shown to the user.\n\n"
                        f"Question: {question}"
                    )
                },
            ]
        )
        hypothetical_answer = hyp_resp.choices[0].message.content
        docs = self._retrieve(hypothetical_answer)
        context = self._format_context(docs)
        
        # generate answer from the generated answer 
        resp = self._chat([
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ])
        return resp.choices[0].message.content
    