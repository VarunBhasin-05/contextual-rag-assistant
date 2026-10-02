# Contextual RAG Document Assistant



A high-performance, grounded Retrieval-Augmented Generation (RAG) microservice built with **FastAPI**, **LangChain**, and **ChromaDB**.



The system enables users to upload unstructured documents (PDFs, TXT) and query them conversationally. It guarantees factual grounding by providing page-level citations and enforces zero-hallucination guardrails when context is unavailable.



---



## 🏛️ Architecture Overview



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

```

---



## 🛠️ Installation & Setup Guide


1. Prerequisites:

   

   Python 3.10+ installed on your system.

   

   A free Groq Cloud API Key from console.groq.com.



3. Clone the Repository:



   git clone https://github.com/VarunBhasin-05/contextual-rag-assistant.git



   cd contextual-rag-assistant



4. Create & Activate Virtual Environment:



   Windows (PowerShell):



   python -m venv venv



   venv\Scripts\activate



   macOS / Linux:



   python3 -m venv venv



   source venv/bin/activate



5. Install Dependencies:



   pip install --upgrade pip



   pip install -r requirements.txt



   pip install sentence-transformers python-multipart



6. Configure API Key:



   Create a .env file in the project root:



   Code snippet: GROQ_API_KEY=gsk_your_actual_groq_api_key_here



7. Run the Application:



   uvicorn app.main:app --reload --port 8000

   

8. Interactive API Docs

  

   Once running, navigate to: http://127.0.0.1:8000/docs



---

## 📡 API Endpoints




Ingest Document



Endpoint: POST /upload



Payload: multipart/form-data (.pdf or .txt)



Ask Question



Endpoint: POST /ask



Request Body:



JSON

{

  "question": "What are the primary findings in the document?"

}

Response: Grounded answer with source file, page number, and snippet citations.


---
