import os
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

CHROMA_PERSIST_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

# sentence-transformers/all-MiniLM-L6-v2 runs locally and is free
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def reset_vector_store():
    store = get_vector_store()
    store.delete_collection()

def get_vector_store() -> Chroma:
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_function,
        collection_name="rag_knowledge_base"
    )

def add_documents_to_store(chunks: List[Document]):
    store = get_vector_store()
    store.add_documents(chunks)
    return len(chunks)

def query_similar_chunks(query: str, k: int = 4) -> List[Document]:
    store = get_vector_store()
    return store.similarity_search(query, k=k)