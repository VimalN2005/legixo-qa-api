# LangGraph Implementation details

This document outlines the LangGraph design and execution structure for the Legixo Q&A HTTP API.

## State Definition
The agent's state is modeled using Python's `TypedDict`. It tracks the question, query, retrieved chunks, relevant chunks, current answers, citations, and attempt count.

```python
from typing import List, TypedDict, Dict, Any

class AgentState(TypedDict):
    question: str                   # Original user query
    current_query: str             # Search query (modified if query rewriting occurs)
    retrieved_chunks: List[Dict]    # Unfiltered search matches from Pinecone
    relevant_chunks: List[Dict]     # Filtered chunks graded as relevant by the LLM Grader
    answer: str                     # Grounded generated answer or fallback response
    citations: List[str]            # Unique source file names cited
    search_attempts: int            # Iteration tracker for loop prevention
    max_attempts: int               # Limit on number of rewrites (e.g. 2)
    trace: List[str]                # Execution log of node actions
```

## Graph Nodes

### 1. `retrieve`
* **Purpose**: Query Pinecone using the `current_query` vector embedding.
* **Logic**: Uses the configured embedding model (`models/text-embedding-004` for Google or `text-embedding-3-small` for OpenAI) to embed the query. Queries Pinecone to retrieve the top 4 matching document chunks.
* **Output**: Updates `retrieved_chunks` and logs the action in `trace`.

### 2. `grade_documents`
* **Purpose**: Re-evaluate retrieved chunks to ensure they are actually useful for answering the user's question, filtering out noise.
* **Logic**: Uses structured output parsing (`with_structured_output(GradeDocument)`) to ask the LLM to return `True` or `False` for each chunk's relevance to the original `question`.
* **Output**: Updates `relevant_chunks` with only the relevant chunks and logs graded actions.

### 3. `rewrite_query`
* **Purpose**: Rewrite search query if no relevant chunks are found.
* **Logic**: Asks the LLM to synthesize a new search query using the original question and the previously failed query.
* **Output**: Updates `current_query` and increments `search_attempts`.

### 4. `generate_answer`
* **Purpose**: Produce a factually grounded answer with citations.
* **Logic**: Asks the LLM to answer the question using *only* the contents of `relevant_chunks`. The LLM is directed to decline to answer if the information is missing. Unique filenames are extracted into `citations`.
* **Output**: Updates `answer` and `citations`.

### 5. `no_info`
* **Purpose**: Fallback endpoint when search loops terminate without results.
* **Logic**: Assigns a standard "cannot find" answer and clears citations.
* **Output**: Updates `answer` and `citations`.

---

## Routing & Loops
Conditional edges originate from the grading node:

* **Route to `generate_answer`**: If `len(relevant_chunks) > 0`, go directly to generation.
* **Route to `rewrite_query`**: If `relevant_chunks` is empty but `search_attempts` < `max_attempts`, attempt query expansion.
* **Route to `no_info`**: If `relevant_chunks` is empty and `search_attempts` >= `max_attempts`, route directly to fallback termination.
