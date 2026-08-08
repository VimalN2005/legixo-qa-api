from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from graph import graph_app
from ingest import run_ingestion

app = FastAPI(
    title="Legixo Legal Q&A API", 
    description="A microservice for Q&A over legal notes using FastAPI, LangGraph, and Pinecone.",
    version="1.0.0"
)

# Pydantic Request/Response validation
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    citations: List[str]
    trace: List[str]

@app.get("/")
def read_root():
    return {
        "name": "Legixo Legal Q&A HTTP API",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "POST /ask": "Submit a question. Accepts JSON {'question': 'string'}.",
            "POST /ingest": "Trigger or re-trigger corpus ingestion."
        }
    }

@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    # Check if index exists or keys are present
    if not os.getenv("PINECONE_API_KEY"):
        raise HTTPException(status_code=500, detail="PINECONE_API_KEY not configured on server.")
    
    # Initialize the LangGraph State
    initial_state = {
        "question": request.question,
        "current_query": request.question,
        "retrieved_chunks": [],
        "relevant_chunks": [],
        "answer": "",
        "citations": [],
        "search_attempts": 0,
        "max_attempts": 2, # Limits loops/rewriting to prevent infinite spin
        "trace": ["Initialized state."]
    }
    
    try:
        # Execute StateGraph
        output_state = graph_app.invoke(initial_state)
        
        return AskResponse(
            answer=output_state.get("answer", "I cannot find the answer in the provided documents."),
            citations=output_state.get("citations", []),
            trace=output_state.get("trace", [])
        )
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Execution error within LangGraph: {str(e)}\n{error_details}"
        )

@app.post("/ingest")
def trigger_ingest():
    try:
        run_ingestion()
        return {"status": "success", "message": "Corpus ingestion completed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion process failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    print(f"Starting server on {host}:{port}...")
    uvicorn.run(app, host=host, port=port)
