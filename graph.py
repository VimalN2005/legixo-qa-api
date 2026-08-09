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
class ChunkRelevance(BaseModel):
    chunk_id: str = Field(description="The unique ID of the document chunk.")
    relevant: bool = Field(description="Is this chunk relevant to the query? True or False.")

class GradeDocumentsResponse(BaseModel):
    grades: List[ChunkRelevance] = Field(description="Relevance grades for all provided chunks.")

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
        # Using gemini-2.0-flash for speed and reliability
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
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
    """Grade all retrieved chunks for relevance to the original question in a single batch call."""
    question = state["question"]
    chunks = state["retrieved_chunks"]
    trace = list(state.get("trace", []))
    
    if not chunks:
        trace.append("grade_documents: No chunks to grade.")
        return {"relevant_chunks": [], "trace": trace}
        
    llm = get_llm()
    # Batch grader structure
    grader = llm.with_structured_output(GradeDocumentsResponse)
    
    # Format chunks for LLM review
    chunks_str = ""
    for chunk in chunks:
        chunks_str += f"--- Chunk ID: {chunk['chunk_id']} ---\n{chunk['text']}\n\n"
        
    system_prompt = (
        "You are a legal assistant grading the relevance of retrieved document chunks to a user query. "
        "Review each chunk and determine if it contains facts or context relevant to answering the query. "
        "Return a relevance grade (true or false) for every single chunk by ID."
    )
    user_prompt = f"Query: {question}\n\nDocument Chunks:\n{chunks_str}"
    
    relevant_chunks = []
    try:
        result = grader.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        # Create a lookup for relevance
        relevance_map = {item.chunk_id: item.relevant for item in result.grades}
        
        for chunk in chunks:
            is_relevant = relevance_map.get(chunk["chunk_id"], False)
            if is_relevant:
                relevant_chunks.append(chunk)
                trace.append(f"grade_documents: Chunk '{chunk['chunk_id']}' graded RELEVANT.")
            else:
                trace.append(f"grade_documents: Chunk '{chunk['chunk_id']}' graded NOT RELEVANT.")
                
    except Exception as e:
        trace.append(f"grade_documents: Error in batch grading: {e}")
        # Fallback: if batch grading fails, assume all chunks are relevant to be safe
        relevant_chunks = chunks
        trace.append("grade_documents: Falling back to treating all chunks as relevant due to error.")
            
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
        "4. Be clear, professional, and factual.\n"
        "5. Be comprehensive: Include all specific details, conditions, requirements, or related obligations (such as returning property or listing counter-arguments) mentioned in the relevant section of the document chunks."
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
