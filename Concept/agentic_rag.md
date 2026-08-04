# AGENTIC RAG PIPELINE 

### Source: https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/Agentic_RAG.ipynb

- Traditional RAG systems take queries as-is, often leading to poor retrievals for ambiguous, context-lacking, or complex queries. 
- Agentic RAG intelligently preprocesses queries to bridge this gap. In the query path, the primary agentic step is query reformulation, comprising multi-turn, query expansion, or query decomposition. 
- This query reformulation step is critical to obtaining the most robust RAG results, and is one component of a system engineered to generate the most accurate query responses.

- In query reformulation, context is added or queries are restructured from the original input: for multi-turn, adding iterative dialogue context; for query expansion, adding additional context to help a short query return optimal results; for query decomposition, taking complex multi-faceted queries that require reasoning across several unrelated documents, and breaking them down into several sub-queries that help obtain the most relevant retrievals. This agentic component handles all of this reformulation autonomously, augmenting the user's query to help obtain the response they need.