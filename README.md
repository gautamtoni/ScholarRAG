# ScholarRAG – Retrieval-Augmented Academic QA System

ScholarRAG is an AI-powered academic assistant that answers questions from PDF documents using Retrieval-Augmented Generation (RAG).

## 🚀 Features

- Semantic search using vector embeddings
- Session-based chat memory
- FastAPI backend
- Streamlit frontend
- Modular and scalable architecture

## 🏗️ Architecture

User → Streamlit → FastAPI → ChromaDB → Groq LLaMA → Response

## 🧠 Tech Stack

- FastAPI
- Streamlit
- LangChain
- ChromaDB
- Sentence-Transformers (MiniLM)
- Groq LLaMA 3.1
- uv package manager

## ⚙️ Setup Instructions

### 1. Install dependencies

```bash
uv add fastapi uvicorn streamlit langchain langchain-community langchain-core langchain-groq chromadb sentence-transformers pypdf python-dotenv requests python-multipart
