# 🤖 AI Document Assistant (RAG System)

An AI-powered document assistant that allows users to **chat with PDFs using Retrieval-Augmented Generation (RAG)**.

This project uses **LangChain, FAISS, and a local LLM (Ollama)** to retrieve relevant information from documents and generate answers.

---

## 🚀 Features

- Upload and analyze multiple PDFs
- Ask questions about document content
- Semantic search using vector embeddings
- Retrieval-Augmented Generation (RAG)
- Local LLM inference using Ollama
- Voice input support
- Chat-style interface

---

## 🧠 How It Works

1. User uploads PDF documents
2. Documents are split into chunks
3. Text chunks are converted into embeddings
4. Embeddings are stored in a FAISS vector database
5. User asks a question
6. Relevant document chunks are retrieved
7. Context is sent to the LLM
8. AI generates the final answer

---

## 🛠 Tech Stack

- Python
- Streamlit
- LangChain
- FAISS Vector Database
- HuggingFace Embeddings
- Ollama (Local LLM)
- Faster Whisper (Speech Recognition)

---

## 📂 Project Structure
