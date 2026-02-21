import os
import uuid
from typing import List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from backend.pdf_reader import extract_text_from_pdf
from backend.processor import chunk_texts
from backend.embeddings import EmbeddingManager
from backend.vectorstore import FaissVectorStore

# -------------------- Google Gemini ----------------------
import google.generativeai as genai

# -------------------- Load Environment -------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)     # Configure Gemini API 

# -------------------- App Setup --------------------------
app = FastAPI(title="DocuMind Backend")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Initialize Embeddings & Vectorstore --------------------
embedding_manager = EmbeddingManager()
EMB_DIM = embedding_manager.model.get_sentence_embedding_dimension()
vector_store = FaissVectorStore(dim=EMB_DIM)

UPLOADED = {}  # store metadata of uploaded files

# -------------------- Helper: Gemini Chat -------------------------------------
def get_gemini_response(prompt: str) -> str:
    """Send prompt to Google Gemini and return response text."""
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# -------------------- Endpoints -----------------------------------------------

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a single PDF, split into chunks, generate embeddings, and store metadata."""
    os.makedirs("uploaded_pdfs", exist_ok=True)
    file_id = str(uuid.uuid4())
    path = os.path.join("uploaded_pdfs", f"{file_id}.pdf")

    # Save uploaded file
    with open(path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Extract text and create embeddings
    pages = extract_text_from_pdf(path)
    chunks = chunk_texts(pages)
    texts = [c["chunk"] for c in chunks]
    embs = embedding_manager.embed_texts(texts)

    # Prepare metadata for each chunk
    metas = []
    for c in chunks:
        metas.append({
            "file_id": file_id,
            "file_name": file.filename,
            "chunk_id": c["id"],
            "page": c["page"],
            "start": c["start"],
            "end": c["end"],
            "text_preview": c["chunk"][:250]
        })

    vector_store.add(embs, metas) # Add to vector store

    # Track uploaded file
    UPLOADED[file_id] = {
        "filename": file.filename,
        "path": path,
        "num_chunks": len(chunks)
    }

    return {
        "file_id": file_id,
        "filename": file.filename,
        "num_chunks": len(chunks)
    }


@app.post("/upload_pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Upload multiple PDFs at once, process each and return metadata."""
    os.makedirs("uploaded_pdfs", exist_ok=True)
    uploaded = []

    for file in files:
        file_id = str(uuid.uuid4())
        path = os.path.join("uploaded_pdfs", f"{file_id}.pdf")
        with open(path, "wb") as f:
            content = await file.read()
            f.write(content)

        pages = extract_text_from_pdf(path)
        chunks = chunk_texts(pages)
        texts = [c["chunk"] for c in chunks]
        embs = embedding_manager.embed_texts(texts)

        metas = []
        for c in chunks:
            metas.append({
                "file_id": file_id,
                "file_name": file.filename,
                "chunk_id": c["id"],
                "page": c["page"],
                "start": c["start"],
                "end": c["end"],
                "text_preview": c["chunk"][:250]
            })

        vector_store.add(embs, metas)

        UPLOADED[file_id] = {
            "filename": file.filename,
            "path": path,
            "num_chunks": len(chunks)
        }

        uploaded.append({
            "file_id": file_id,
            "filename": file.filename,
            "num_chunks": len(chunks)
        })

    return {"uploaded": uploaded}


@app.post("/query")
async def query_doc(query: str = Form(...), top_k: int = Form(5)):
    """Query  uploaded PDFs using embeddings and return answer with citations."""
    # Generate embedding for the query
    q_emb = embedding_manager.embed_texts([query])[0]
    results = vector_store.query(q_emb, top_k=top_k)

    if not results:
        return {"error": "No relevant chunks found for your query.", "citations": []}

    # Build context and track citations
    context_texts = []
    citations = []
    seen_chunks = set()

    for i, (meta, score) in enumerate(results):
        if meta["chunk_id"] in seen_chunks:
            continue
        seen_chunks.add(meta["chunk_id"])

        filename = UPLOADED.get(meta["file_id"], {}).get("filename", "Unknown File")

        context_texts.append(
            f"[{i+1}] {meta['text_preview'][:250]}"
        )

        citations.append({
            "rank": i+1,
            "source": f"{filename} (ID: {meta['file_id']})",
            "page": meta["page"],
            "score": float(score)
        })

    # Prepare prompt for Gemini with context
    prompt = (
        "You are DocuMind — an intelligent assistant that answers questions based on PDF content.\n"
        "Use the provided CONTEXT to answer concisely (3–5 lines max).\n"
        "When referencing a source, include only the page number in parentheses (e.g., [Page 8]).\n"
        "Do not include file IDs or long UUIDs in the answer.\n\n"
        f"CONTEXT:\n{chr(10).join(context_texts)}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer clearly and factually."
    )

    # Get answer from Gemini
    try:
        answer = get_gemini_response(prompt)
    except Exception as e:
        # Fallback in case LLM call fails
        answer = "LLM call failed: " + str(e) + "\n\nContext provided:\n" + "\n".join(context_texts)

    return {"answer": answer, "citations": citations}