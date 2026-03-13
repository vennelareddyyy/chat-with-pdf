# AI Document Assistant (RAG System)

This project allows users to chat with PDFs using Retrieval-Augmented Generation (RAG).

## Features
- Upload multiple PDFs
- Ask questions about documents
- Semantic search using vector embeddings
- Local LLM inference using Ollama
- Voice input support
- Conversation memory

## Tech Stack
- Python
- Streamlit
- LangChain
- FAISS Vector Database
- HuggingFace Embeddings
- Ollama (Local LLM)
- Faster Whisper (Speech Recognition)

## How it Works
1. PDFs are uploaded
2. Documents are split into chunks
3. Embeddings are generated
4. Stored in FAISS vector database
5. User query retrieves relevant chunks
6. Context sent to LLM for final answer

## Run the Project

Install dependencies:

pip install -r requirements.txt

Run the app:

streamlit run app.py