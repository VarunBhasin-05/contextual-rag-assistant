from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_document(file_path: str) -> List[Document]:
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(file_path)
    elif path.suffix.lower() in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    return loader.load()

def chunk_documents(docs: List[Document], chunk_size: int = 600, chunk_overlap: int = 100) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(docs)