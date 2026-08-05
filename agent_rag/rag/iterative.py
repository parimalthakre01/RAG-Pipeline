from rag.base import BaseRAG
import json
# Iterative RAG works by giving the LLM tools and letting it drive a loop — it decides what to search, checks if the results are enough, and keeps going until it has a complete answer.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the legal knowledge base for relevant documents. "
                "Use targeted, sp question verbatim. "
                "Call multiple times with different queries to cover all aspects."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Specific search query"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (1-10)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_context",
            "description": (
                "Evaluate whether the retrieved context is sufficient to answer "
                "the question. Caund before deciding "
                "whether to search again or answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": ["sufficient", "needs_more", "unanswerable"],
                        "description": (
                            "sufficient   – ready to answer\n"
                            "needs_more   – gaps remain, will search again\n"
                            "unan the knowledge base"
                        ),
                    },
                    "gaps": {
                        "type": "string",
                        "description": "What is still missing (required when verdict is needs_more)",
                    },
                },
                "required": ["verdict"],
            },
        },
    },
]

class interativeRag(BaseRAG):
    def run(self, question: str, max_rounds : int = 5) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a legal research assistant with access to a knowledge base.\n\n"
                    "Process:\n"
                    "1. Break the question into specific information needs\n"
                    "2. Call search_knowledge_base with a targeted query\n"
                    "3. Call evaluate_context to check if you have enough\n"
                    "4. If gaps remain, search again with a refined query\n"
                    "5. Once context is sufficient, write the final answer\n\n"
                    "Rules:\n"
                    "- Answer only from retrieved context, never from prior knowledge\n"
                    "- Always cite the source document\n"
                    f"- Maximum search rounds: {max_rounds}"
                ),
            },
            {"role": "user", "content": question},
        ]
        
        rounds = 0
        
        while rounds < max_rounds:
            resp = self._chat(messages, tools=TOOLS)
            msg = resp.choices[0].message
            
            if resp.choices[0].finish_reason == "stop":
                return msg.content
            
            # LLM wants to call tools
            if resp.choices[0].finish_reason != "tool_calls":
                break
            messages.append(msg)
            
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments)
                
                if call.function.name == "search_knowledge_base":
                    docs = self._retrieve(args["query"], top_k=args.get("top_k", 5))
                    result = self._format_context(docs) if docs else "No results found."
                    rounds += 1
                    
                elif call.function.name == "evaluate_context":
                    verdict = args["verdict"]
                    gaps = args.get("gaps", "")
                    result = f"Verdict recorded: {verdict}. {gaps}".strip()

                else:
                    result = f"Unknown tool: {call.function.name}"
                    
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })
        messages.append({
            "role": "user",
            "content": "You have reached the search limit. Answer using the information gathered so far.",
        })
        resp = self._chat(messages)
        return resp.choices[0].message.content