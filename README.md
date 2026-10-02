# Contextual RAG Document Assistant

A high-performance, grounded Retrieval-Augmented Generation (RAG) microservice built with **FastAPI**, **LangChain**, and **ChromaDB**. 

The system enables users to upload unstructured documents (PDFs, TXT) and query them conversationally. It guarantees factual grounding by providing page-level citations and enforces zero-hallucination guardrails when context is unavailable.

---

## Architecture Overview

```text
[ Document Ingestion ]
PDF / TXT ---> PyPDFLoader ---> Recursive Chunking (chunk_size=600, overlap=100)
                                      │
                                      ▼
[ Vector Store ]          Embeddings: sentence-transformers/all-MiniLM-L6-v2
ChromaDB (Local persistent) <─────────┘

[ Retrieval & Generation ]
User Query ---> Semantic Search (Top-k Chunks + Source Metadata)
                       │
                       ▼
Prompt Template [Grounded Context + Query] ---> Groq LLM (Qwen 3.8 27B)
                       │
                       ▼
Structured Response + Page-Level Citations (Source, Page, Snippet)
