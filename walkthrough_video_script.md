# 📹 Video Walkthrough Script: Legixo Legal Q&A HTTP API
**Target Duration**: 5 to 8 minutes
**Speaker**: Vimal Sahani

---

## 🛠️ Step 0: Prep Before Recording
1. Open **VS Code** with the active workspace set to: `legixo-qa-api/`.
2. Open these files in tabs to click through:
   - `README.md`
   - `ingest.py`
   - `graph.py`
   - `main.py`
   - `docs/langgraph.md` (to show the diagram)
3. Open a **terminal** inside VS Code.
4. Have your **Pinecone Console** and **FastAPI Swagger UI** (`http://127.0.0.1:8000/docs`) open in Chrome.

---

## 🎬 Section 1: Intro (0:00 - 0:45)
* **Visual**: Show your webcam or point to `README.md` in VS Code.
* **Speaker**:
> "Hi everyone, my name is Vimal Sahani. Today, I'm excited to walk you through my submission for the Legixo Thinklabs AI Intern take-home assignment: the **Agentic Legal Q&A HTTP API**.
> 
> The goal of this project is to build an API that answers legal questions using *only* a provided set of documents, returning clean citations and an execution trace, and gracefully falling back when the information is not present.
> 
> My solution is built in Python 3.14 using **FastAPI**, **LangGraph** for the agent state workflows, and **Pinecone** as the vector database, supporting both Google Gemini and OpenAI model backends."

---

## 💾 Section 2: Ingestion & Pinecone Setup (0:45 - 2:00)
* **Visual**: Click on [ingest.py](file:///C:/Users/LENOVO/.gemini/antigravity/scratch/legixo-qa-api/ingest.py). Scroll through the code.
* **Speaker**:
> "Let's start with document ingestion in `ingest.py`. 
> 
> The script reads legal notes from the `gen_ai_takehome_sample_corpus/` directory. It uses LangChain's `RecursiveCharacterTextSplitter` to segment files into semantically coherent text chunks.
> 
> To ensure **idempotency** (avoiding double-ingest duplication issues):
> 1. It initializes the Pinecone connection. If the index exists but has a different dimension—for example, if you switch LLM providers from Google (768 dims) to OpenAI (1536 dims)—it deletes and recreates the index automatically.
> 2. It flushes existing vectors inside the namespace using `index.delete(delete_all=True)` right before upserting the new embeddings.
> 
> Let's run the ingestion now."
* **Action**: Run `python ingest.py` in the terminal. Show the success output.

---

## 🕸️ Section 3: LangGraph Agent Design (2:00 - 4:00)
* **Visual**: Open [docs/langgraph.md](file:///C:/Users/LENOVO/.gemini/antigravity/scratch/legixo-qa-api/docs/langgraph.md) to show the flow diagram, then switch to [graph.py](file:///C:/Users/LENOVO/.gemini/antigravity/scratch/legixo-qa-api/graph.py).
* **Speaker**:
> "Now let's look at the core of the system: the LangGraph StateGraph in `graph.py`.
> 
> Instead of using a single LLM call, we run a state-driven loop. The graph defines 5 key nodes:
> 1. `retrieve`: Queries Pinecone using the current query vector.
> 2. `grade_documents`: Filters chunks using an LLM evaluator to check if they are actually relevant.
> 3. `rewrite_query`: If no relevant chunks are found, it uses the LLM to rewrite and broaden the search query.
> 4. `generate_answer`: Formulates the final grounded answer with citations using only the graded relevant chunks.
> 5. `no_info`: If we exceed 2 search iterations without results, this node triggers a clean refusal.
> 
> Let's look at the routing edge `decide_to_generate`. It routes to `generate_answer` if we have relevant chunks, loops back to `rewrite_query` if not, and terminates at `no_info` once `search_attempts` hits the safety limit. This prevents infinite looping."

---

## ⚡ Section 4: Live API & Swagger Demo (4:00 - 6:00)
* **Visual**: Open [main.py](file:///C:/Users/LENOVO/.gemini/antigravity/scratch/legixo-qa-api/main.py) briefly, run the server `python main.py`, and switch to Chrome displaying `http://127.0.0.1:8000/docs`.
* **Speaker**:
> "The API is served via FastAPI in `main.py`. It exposes a `POST /ask` endpoint for questions, and a `POST /ingest` endpoint.
> 
> Let's test the `/ask` endpoint live using the Swagger UI.
> 
> First, let's ask a valid in-corpus question: *'What notice period applies when Bluecrest or Priya Nambiar ends the employment agreement?'*"
* **Action**: Submit the query. Scroll to the response.
* **Speaker**:
> "As you can see, the API returns the correct answer ('60 days written notice'), cites `02_employment_agreement_excerpt.md`, and includes a full `trace` of nodes executed: `retrieve` -> `grade_documents` -> `generate_answer`.
> 
> Now, let's test an out-of-corpus question that isn't mentioned in the documents: *'What is the capital of France?'*"
* **Action**: Submit the out-of-corpus query.
* **Speaker**:
> "Here, the grader flags the retrieved documents as irrelevant. The graph routes to `rewrite_query` to broaden the query and retrieve again. Since the new chunks are still irrelevant, it terminates gracefully in the `no_info` fallback, returning: *'I cannot find the answer in the provided documents.'* with no citations, proving it doesn't hallucinate."

---

## 📊 Section 5: Automated Evaluation Suite (6:00 - End)
* **Visual**: Switch to terminal. Run `python run_tests.py`. Then open `eval_results.md`.
* **Speaker**:
> "To guarantee quality, I built an evaluation suite in `run_tests.py` testing 15 total cases. Let's run it.
> 
> The script calls the API for each test case, validates the returned citations and key grounding facts, and saves the results in `eval_results.md`.
> 
> As you can see, our agent scores a perfect 15 out of 15!
> 
> That concludes my walk-through. All the code, dependencies, and configuration steps are fully documented in the README. Thank you for your time, and I look forward to your feedback!"
