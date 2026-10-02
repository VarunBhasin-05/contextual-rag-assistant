import os
from dotenv import load_dotenv
load_dotenv()
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.vectorstore import query_similar_chunks

# Set your API key in environment or .env file
# export GROQ_API_KEY="gsk_..."
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0.1,
    api_key="GROQ_API_KEY"
)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly based on the provided context.
If the answer cannot be found in the context, say: "I do not have enough information in the provided documents to answer that."

Context:
{context}
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])

def answer_query(query: str) -> Dict[str, Any]:
    relevant_docs = query_similar_chunks(query, k=4)
    
    # Build context string and structured citations
    context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
    citations = [
        {
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", 0) + 1,
            "snippet": doc.page_content[:150] + "..."
        }
        for doc in relevant_docs
    ]

    chain = PROMPT_TEMPLATE | llm
    response = chain.invoke({"context": context_text, "question": query})

    return {
        "answer": response.content,
        "citations": citations
    }