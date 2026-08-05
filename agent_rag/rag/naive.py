from rag.base import BaseRAG


class NaiveRAG(BaseRAG):
    """
    Single retrieval pass.
    embed(question) → retrieve → generate.
    Fast but struggles with abstract queries or multi-part questions.
    """

    def run(self, question: str) -> str:
        docs = self._retrieve(question)
        context = self._format_context(docs)
        resp = self._chat([
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ])
        return resp.choices[0].message.content
