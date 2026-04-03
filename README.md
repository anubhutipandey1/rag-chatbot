\# RAG Chatbot



A document Q\&A chatbot built from scratch to understand RAG architecture hands-on.



\## What it does

Upload a PDF, ask questions, get grounded answers with source citations.



\## Pipeline

\- Stage 1: PDF ingestion → chunking → embeddings → ChromaDB

\- Stage 2: Query retrieval via semantic similarity search

\- Stage 3: Answer generation using Llama 3.2 (runs locally via Ollama)

\- Stage 4: Reranking with CrossEncoder for improved retrieval quality

\- Stage 5: Evaluation using LLM-as-a-judge



\## Stack

\- ChromaDB (vector database)

\- sentence-transformers (embedding model)

\- CrossEncoder (reranker)

\- Ollama + Llama 3.2 (local LLM)

\- Python



\## Setup

pip install chromadb pymupdf sentence-transformers ollama

ollama pull llama3.2



\## Key finding

500-word chunks outperformed 200-word chunks for this document type.

Faithfulness: 0.77 vs 0.33. Chunk size is document-dependent.

