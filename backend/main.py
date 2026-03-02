import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config import config

load_dotenv()

app = FastAPI(title="ScholarRAG – Retrieval-Augmented Academic QA System",description="An AI-powered academic assistant using RAG, ChromaDB, and Groq LLaMA.",version="1.0.0")

# ---- API KEY ----
if "GROQ_API_KEY" not in os.environ:
    raise ValueError("GROQ_API_KEY not found in .env file")

# ---- Load Embeddings ----
embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL_NAME
)

# ---- Load Vector DB ----
vectordb = Chroma(
    persist_directory=config.CHROMA_PERSIST_DIRECTORY,
    embedding_function=embeddings
)

# ---- Load LLM ----
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=300,
)

# ---- Simple Memory Store ----
chat_memory = {}  # {session_id: [messages]}
MAX_HISTORY = 6   # last 3 user+assistant pairs


class QueryRequest(BaseModel):
    question: str
    session_id: str


@app.post("/ask")
def ask_question(request: QueryRequest):

    query = request.question
    session_id = request.session_id

    # Initialize memory if new session
    if session_id not in chat_memory:
        chat_memory[session_id] = []

    # ---- Retrieve Documents ----
    docs = vectordb.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # ---- System Prompt ----
    system_prompt = """
You are an intelligent assistant.

1. If the question refers to previous conversation,
   answer using chat history.

2. If the question is academic,
   use the provided document context.

3. If answer not found anywhere, say you don't know.

Keep answers concise (max 3 sentences).
"""

    messages = [SystemMessage(content=system_prompt)]

    # Add previous history (windowed)
    history = chat_memory[session_id][-MAX_HISTORY:]
    messages.extend(history)

    # Add current question
    messages.append(
        HumanMessage(
            content=f"Context:\n{context}\n\nQuestion:\n{query}"
        )
    )

    # ---- Model Call ----
    response = model.invoke(messages)

    # ---- Save Memory ----
    chat_memory[session_id].append(HumanMessage(content=query))
    chat_memory[session_id].append(AIMessage(content=response.content))

    # ---- Extract Sources ----
    sources = list(
        set(
            [
                f"{doc.metadata.get('source')} (Page {doc.metadata.get('page')})"
                for doc in docs
            ]
        )
    )

    return {
        "answer": response.content,
        "sources": sources
    }
