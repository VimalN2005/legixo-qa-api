# Legixo Q&A HTTP API (Gen AI Intern Take-Home)

A production-ready Q&A HTTP API built over fictional legal-style notes.

## Tech Stack
* **Language:** Python 3.10+ (tested on Python 3.14.2)
* **Framework:** FastAPI & Uvicorn
* **RAG Orchestrator:** LangGraph StateGraph (for modular nodes, loops, and query rewriting)
* **Vector Database:** Pinecone
* **LLM Providers:** Flexible support for Google Gemini (recommended) or OpenAI

---

## Architecture and LangGraph Flow

This API runs a structured agent flow using LangGraph. The graph includes a query rewriter and document grader to loop and refine search terms if initial results are insufficient, with a strict safety-limit on iterations.

```
                  ┌──────────────────────┐
                  │  Start: User Query   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
            ┌────►│ Node: Retrieve       │◄────┐
            │     │ (Search Pinecone)    │     │
            │     └──────────┬───────────┘     │
            │                │                 │
            │                ▼                 │
            │     ┌──────────────────────┐     │
            │     │ Node: Grade Chunks   │     │ (Query loop)
            │     │ (Relevance check)    │     │
            │     └──────────┬───────────┘     │
            │                │                 │
            │                ▼                 │
            │      /───────────────────\       │
            │     <   Any relevant      >      │
            │      \  chunks found?    /       │
            │        /               \         │
            │       / Yes             \ No     │
            │      ▼                   ▼       │
            │  ┌───────────┐    /─────────────\│
            │  │  Generate │   < Attempts <   >│
            │  │  Answer   │    \ Max (2)?   / │
            │  └─────┬─────┘      /        \   │
            │        │           / Yes      \No│
            │        │          ▼            ▼
            │        │   ┌──────────────┐ ┌──────────┐
            │        │   │ Rewrite Query│ │ No Info  │
            │        │   └──────┬───────┘ └────┬─────┘
            │        │          └──────────────┘
            │        │ (State update)
            ▼        ▼
       ┌──────────────────────┐
       │   End: JSON Response │
       └──────────────────────┘
```

For more detailed information, see [`docs/langgraph.md`](docs/langgraph.md).

---

## Installation & Setup

1. **Clone/Navigate to the Repo:**
   ```bash
   cd legixo-qa-api
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.10+ installed. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Copy the template `.env.example` to `.env`:
   ```bash
   copy .env.example .env
   ```
   Open `.env` and fill in your keys:
   * `PINECONE_API_KEY`: Your Pinecone API key.
   * `LLM_PROVIDER`: Set to `google` to use Google Gemini, or `openai` to use OpenAI.
   * `GOOGLE_API_KEY`: Required if provider is `google`.
   * `OPENAI_API_KEY`: Required if provider is `openai`.

---

## Document Ingestion

The legal corpus zip is extracted into the `gen_ai_takehome_sample_corpus/` directory.

To load, chunk, embed, and upload the documents to Pinecone, run:
```bash
python ingest.py
```
Or you can trigger it via an API call once the server is running (see below).

### What happens if you run ingest twice?
Ingest is designed to be **fully idempotent**:
1. When run, the script connects to Pinecone and checks if the index already exists.
2. If the index exists but the dimensions mismatch (e.g. you switched from Google's 768-dim embeddings to OpenAI's 1536-dim embeddings), the script will delete and recreate the index.
3. The script clears the index (`index.delete(delete_all=True)`) before uploading new chunks, ensuring no duplicate vectors or stale documents are left.

---

## Running the API Server

Start the FastAPI application using Uvicorn:
```bash
python main.py
```
The server will start by default on `http://127.0.0.1:8000`.

---

## REST Endpoints and Usage

### 1. Ingestion (`POST /ingest`)
Trigger document chunking and indexing via API.
* **cURL Request:**
  ```bash
  curl -X POST http://127.0.0.1:8000/ingest
  ```
* **Response:**
  ```json
  {
    "status": "success",
    "message": "Corpus ingestion completed successfully."
  }
  ```

### 2. Q&A Ask (`POST /ask`)
Submit a query to ask questions only from the document set.
* **cURL Request:**
  ```bash
  curl -X POST http://127.0.0.1:8000/ask \
       -H "Content-Type: application/json" \
       -d "{\"question\": \"What notice period applies when Bluecrest or Priya Nambiar ends the employment agreement?\"}"
  ```
* **Sample JSON Response:**
  ```json
  {
    "answer": "When Bluecrest or Priya Nambiar terminates the employment agreement, a written notice period of 60 days applies. During this notice period, the employee must return all company property, including laptops, access badges, and source code access [02_employment_agreement_excerpt.md].",
    "citations": ["02_employment_agreement_excerpt.md"],
    "trace": [
      "Initialized state.",
      "retrieve: Searched for 'What notice period applies when Bluecrest or Priya Nambiar ends the employment agreement?'. Found 4 matches.",
      "grade_documents: Chunk '02_employment_agreement_excerpt.md_chunk_0' graded RELEVANT.",
      "grade_documents: Chunk '02_employment_agreement_excerpt.md_chunk_1' graded NOT RELEVANT.",
      "generate_answer: Successfully generated answer from relevant chunks."
    ]
  }
  ```

* **Fallback / Out-of-Corpus Query:**
  ```bash
  curl -X POST http://127.0.0.1:8000/ask \
       -H "Content-Type: application/json" \
       -d "{\"question\": \"What is the capital of France?\"}"
  ```
* **Response:**
  ```json
  {
    "answer": "I cannot find the answer in the provided documents.",
    "citations": [],
    "trace": [
      "Initialized state.",
      "retrieve: Searched for 'What is the capital of France?'. Found 4 matches.",
      "grade_documents: Chunk '01_matter_memo_arvind_v_northfield.md_chunk_0' graded NOT RELEVANT.",
      "grade_documents: Chunk '03_hearing_notice_template.md_chunk_0' graded NOT RELEVANT.",
      "rewrite_query: Rewrote query to 'capital city of France location' (Attempt 1)",
      "retrieve: Searched for 'capital city of France location'. Found 4 matches.",
      "grade_documents: Chunk '01_matter_memo_arvind_v_northfield.md_chunk_0' graded NOT RELEVANT.",
      "no_info: Max search attempts reached with no relevant chunks. Returning fallback."
    ]
  }
  ```

---

## Evaluation & Self-Testing

We have prepared a self-test suite containing 15 cases (12 in-corpus, 3 out-of-corpus) in `self_test_cases.json`.

To run the automated tests against your running server:
```bash
python run_tests.py
```
This runs each question, validates the returned citations, checks if the answer covers the gold facts, and generates a markdown summary report at `eval_results.md`.

---

## Walkthrough Video & Details
* A video walk-through (setup, ingestion, API querying, out-of-corpus handling, and LangGraph explanation) can be found at: [VIDEO_LINK_HERE]
