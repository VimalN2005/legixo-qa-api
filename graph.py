import os
from typing import List, TypedDict, Dict, Any, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

load_dotenv()

# Define the Graph State
class AgentState(TypedDict):
    question: str
    current_query: str
    retrieved_chunks: List[Dict[str, Any]]
    relevant_chunks: List[Dict[str, Any]]
    answer: str
    citations: List[str]
    search_attempts: int
    max_attempts: int
    trace: List[str]

# Structured output schemas
class GradeDocument(BaseModel):
    relevant: bool = Field(description="Is the document chunk relevant/useful to answer the question? True or False.")

# Helper to get embedding model
def get_embeddings_model():
    provider = os.getenv("LLM_PROVIDER", "google").lower()
    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")

# Helper to get LLM
def get_llm():
    provider = os.getenv("LLM_PROVIDER", "google").lower()
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Using gemini-1.5-flash for speed and reliability
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")

# Nodes
def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """Search Pinecone for relevant documents using current_query."""
    query = state["current_query"]
    trace = list(state.get("trace", []))
    
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "legixo-qa-index")
    
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    
    embeddings = get_embeddings_model()
    query_vector = embeddings.embed_query(query)
    
    # Query Pinecone for top 4 matches
    response = index.query(vector=query_vector, top_k=4, include_metadata=True)
    
    retrieved = []
    for match in response.get("matches", []):
        metadata = match.get("metadata", {})
        retrieved.append({
            "id": match.get("id"),
            "score": match.get("score"),
            "source": metadata.get("source"),
            "text": metadata.get("text"),
            "chunk_id": metadata.get("chunk_id")
        })
        
    trace.append(f"retrieve: Searched for '{query}'. Found {len(retrieved)} matches.")
    return {
        "retrieved_chunks": retrieved,
        "trace": trace
    }

def grade_documents_node(state: AgentState) -> Dict[str, Any]:
    """Grade retrieved chunks for relevance to the original question."""
    question = state["question"]
    chunks = state["retrieved_chunks"]
    trace = list(state.get("trace", []))
    
    llm = get_llm()
    # Structured output ensures we get a strict boolean response
    grader = llm.with_structured_output(GradeDocument)
    
    relevant_chunks = []
    
    for chunk in chunks:
        # Prompt the grader
        system_prompt = (
            "You are a legal assistant grading the relevance of a retrieved document chunk to a user query. "
            "Evaluate if the chunk contains information, facts, or context that is directly relevant to answering the query. "
            "Respond ONLY with the specified schema (relevant: true/false)."
        )
        user_prompt = f"Query: {question}\n\nDocument Chunk:\n{chunk['text']}"
        
        try:
            result = grader.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            if result.relevant:
                relevant_chunks.append(chunk)
                trace.append(f"grade_documents: Chunk '{chunk['chunk_id']}' graded RELEVANT.")
            else:
                trace.append(f"grade_documents: Chunk '{chunk['chunk_id']}' graded NOT RELEVANT.")
        except Exception as e:
            trace.append(f"grade_documents: Error grading chunk '{chunk['chunk_id']}': {e}")
            
    return {
        "relevant_chunks": relevant_chunks,
        "trace": trace
    }

def rewrite_query_node(state: AgentState) -> Dict[str, Any]:
    """If no relevant documents, rewrite the query to broaden the search."""
    question = state["question"]
    attempts = state["search_attempts"]
    trace = list(state.get("trace", []))
    
    llm = get_llm()
    system_prompt = (
        "You are an expert search engine query rewriter. "
        "The previous query failed to retrieve relevant documents for the user's question. "
        "Formulate a different, search-optimized query (keywords, synonyms, or broader concepts) "
        "to find relevant legal document chunks. Return ONLY the raw query string and nothing else."
    )
    user_prompt = f"Original Question: {question}\nFailed Query: {state['current_query']}"
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    new_query = response.content.strip().strip('"').strip("'")
    trace.append(f"rewrite_query: Rewrote query to '{new_query}' (Attempt {attempts + 1})")
    
    return {
        "current_query": new_query,
        "search_attempts": attempts + 1,
        "trace": trace
    }

def generate_answer_node(state: AgentState) -> Dict[str, Any]:
    """Generate answer using ONLY the relevant documents."""
    question = state["question"]
    chunks = state["relevant_chunks"]
    trace = list(state.get("trace", []))
    
    # Prepare prompt context
    context_str = ""
    for i, chunk in enumerate(chunks):
        context_str += f"[Chunk {i+1}] (Source: {chunk['source']})\n{chunk['text']}\n\n"
        
    llm = get_llm()
    
    system_prompt = (
        "You are a strict legal Q&A assistant. Answer the user's question based ONLY on the provided document chunks. "
        "Follow these rules precisely:\n"
        "1. If the provided chunks do not contain the answer, reply with: 'I cannot find the answer in the provided documents.'\n"
        "2. Do not make up facts or use external knowledge.\n"
        "3. Incorporate citations inline or at the end indicating which chunk/source file the information came from (e.g. [01_matter_memo_arvind_v_northfield.md]).\n"
        "4. Be clear, professional, and factual."
    )
    user_prompt = f"Context Chunks:\n{context_str}\nQuestion: {question}"
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    # Extract unique source files as citations
    citations = list(set([chunk["source"] for chunk in chunks]))
    
    trace.append("generate_answer: Successfully generated answer from relevant chunks.")
    return {
        "answer": response.content.strip(),
        "citations": citations,
        "trace": trace
    }

def no_info_node(state: AgentState) -> Dict[str, Any]:
    """Fallback when no relevant documents are found after max attempts."""
    trace = list(state.get("trace", []))
    trace.append("no_info: Max search attempts reached with no relevant chunks. Returning fallback.")
    return {
        "answer": "I cannot find the answer in the provided documents.",
        "citations": [],
        "trace": trace
    }

# Router
def decide_to_generate(state: AgentState) -> Literal["generate_answer", "rewrite_query", "no_info"]:
    """Conditional edge router based on graded documents and attempt counter."""
    if len(state.get("relevant_chunks", [])) > 0:
        return "generate_answer"
    
    if state.get("search_attempts", 0) < state.get("max_attempts", 2):
        return "rewrite_query"
        
    return "no_info"

# Build LangGraph StateGraph
def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("no_info", no_info_node)
    
    # Set Entry Point
    workflow.set_entry_point("retrieve")
    
    # Simple direct edges
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("no_info", END)
    
    # Conditional edge routing from grading
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
            "no_info": "no_info"
        }
    )
    
    return workflow.compile()

# Instantiate the graph
graph_app = build_graph()
