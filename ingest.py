import os
import sys
import glob
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

def get_embeddings_and_dimension():
    provider = os.getenv("LLM_PROVIDER", "google").lower()
    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Error: GOOGLE_API_KEY is not set in environment.")
            sys.exit(1)
        # Using models/gemini-embedding-001
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
        try:
            dummy_vector = embeddings.embed_query("dummy")
            dimension = len(dummy_vector)
            print(f"Detected Gemini embedding model dimension: {dimension}")
        except Exception as e:
            print(f"Error getting embedding dimension: {e}")
            sys.exit(1)
        return embeddings, dimension
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY is not set in environment.")
            sys.exit(1)
        # Using text-embedding-3-small (1536 dimensions)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
        return embeddings, 1536
    else:
        print(f"Error: Unknown LLM_PROVIDER '{provider}'. Must be 'google' or 'openai'.")
        sys.exit(1)

def run_ingestion():
    # 1. Initialize Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Error: PINECONE_API_KEY is not set in environment.")
        sys.exit(1)
    
    index_name = os.getenv("PINECONE_INDEX_NAME", "legixo-qa-index")
    
    pc = Pinecone(api_key=api_key)
    
    # Get embedding model and expected dimension
    embeddings, dimension = get_embeddings_and_dimension()
    print(f"Using embedding provider: {os.getenv('LLM_PROVIDER')} (dimension: {dimension})")
    
    # 2. Check and Create Pinecone Index
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name in existing_indexes:
        # Check if dimension matches
        desc = pc.describe_index(index_name)
        if desc.dimension != dimension:
            print(f"Warning: Index '{index_name}' has dimension {desc.dimension} but we need {dimension}.")
            print("Deleting and recreating index to match embedding dimensions...")
            pc.delete_index(index_name)
            existing_indexes.remove(index_name)
            
    if index_name not in existing_indexes:
        print(f"Creating Pinecone index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Index created successfully.")
    
    index = pc.Index(index_name)
    
    # 3. Clean existing vectors (Double-ingest protection)
    # Deleting all vectors in the namespace to make the ingestion idempotent
    print(f"Clearing index '{index_name}' to prevent duplicate vectors...")
    try:
        index.delete(delete_all=True)
        print("Existing vectors cleared.")
    except Exception as e:
        print(f"Note: Could not clear index (might be empty): {e}")

    # 4. Load and chunk documents
    corpus_dir = os.path.join(os.path.dirname(__file__), "gen_ai_takehome_sample_corpus")
    search_path = os.path.join(corpus_dir, "*.md")
    files = glob.glob(search_path)
    
    if not files:
        print(f"Error: No markdown files found in {corpus_dir}")
        print("Please ensure the corpus is extracted in that folder.")
        sys.exit(1)
        
    print(f"Found {len(files)} markdown files to ingest.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    vectors_to_upsert = []
    
    for file_path in files:
        filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        chunks = text_splitter.split_text(content)
        print(f"Split {filename} into {len(chunks)} chunks.")
        
        # Prepare embeddings
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{filename}_chunk_{i}"
            
            # Generate embedding vector
            vector = embeddings.embed_query(chunk_text)
            
            metadata = {
                "chunk_id": chunk_id,
                "source": filename,
                "text": chunk_text,
                "index": i
            }
            
            vectors_to_upsert.append((chunk_id, vector, metadata))
            
    # 5. Upsert to Pinecone
    print(f"Upserting {len(vectors_to_upsert)} chunks to Pinecone...")
    
    # Pinecone upsert recommends batches of 100 or less
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i+batch_size]
        index.upsert(vectors=batch)
        
    print("Ingestion completed successfully!")

if __name__ == "__main__":
    run_ingestion()
