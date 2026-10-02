import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.ingestion import load_document, chunk_documents
from app.vectorstore import add_documents_to_store, reset_vector_store
from app.rag_engine import answer_query

app = FastAPI(title="Contextual RAG Assistant API", version="1.0.0")

UPLOAD_DIR = Path("data")
UPLOAD_DIR.mkdir(exist_ok=True)

class QueryRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported")
    
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    raw_docs = load_document(str(file_path))
    chunks = chunk_documents(raw_docs)
    total_chunks = add_documents_to_store(chunks)

    return {
        "message": f"Successfully indexed '{file.filename}'",
        "chunks_indexed": total_chunks
    }

@app.post("/ask")
async def ask_question(payload: QueryRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = answer_query(payload.question)
    return result

@app.post("/reset")
def clear_database():
    reset_vector_store()
    return {"message": "Vector store cleared successfully."}

@app.get("/health")
def health():
    return {"status": "healthy"}
