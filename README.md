# RAG Document Q&A Chatbot

A production-deployed Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded in uploaded PDF documents. Built end-to-end — from chunking strategy through vector retrieval, cross-encoder reranking, and LLM generation.

**Live demo:** https://rag-chatbot-anubhuti.streamlit.app

---

## What it does

Upload one or more PDFs, ask questions in natural language, and get answers that cite exactly which document chunks they came from. The system refuses to hallucinate — if the answer isn't in the documents, it says so.

---

## Architecture

```
INGESTION PIPELINE                    QUERY PIPELINE
─────────────────                     ──────────────
PDF upload (PyMuPDF)                  User query (Streamlit)
       ↓                                     ↓
Chunking (200w / 30w overlap)         Semantic retrieval (vector similarity)
       ↓                                     ↓  ← ChromaDB
Embedding (all-MiniLM-L6-v2)         Reranking (cross-encoder/ms-marco)
       ↓                                     ↓
ChromaDB (per-doc collection)         LLM generation (Groq / Llama 3.1 8B)
                                             ↓
                                      Answer + cited source chunks
```

Each uploaded document gets its own ChromaDB collection, enabling selective multi-document search — users choose which documents to include per query.

---

## Key technical decisions

**Chunking: 200 words with 30-word overlap**
Tested against FAQ-style and dense instructional documents. 200-word chunks balance semantic completeness (enough context for the reranker) with retrieval precision (small enough to isolate relevant passages). The 30-word overlap prevents answers from being split across chunk boundaries.

**Two-stage retrieval: embedding + cross-encoder reranking**
Initial vector search (top-k=3 per document) uses cosine similarity on `all-MiniLM-L6-v2` embeddings — fast and scalable. A cross-encoder (`ms-marco-MiniLM-L-6-v2`) then re-scores all retrieved chunks against the query. Cross-encoders are slower but significantly more accurate because they evaluate query and chunk together rather than independently. This pattern — fast retrieval, accurate reranking — is standard in production RAG systems.

**LLM: Groq-hosted Llama 3.1 8B**
Originally built with Ollama (local Llama 3.2). Swapped to Groq for deployment — Groq serves the same Llama model family via API with sub-second latency. The swap required changing only the LLM client initialization; the retrieval and reranking pipeline was untouched. This demonstrates clean architectural separation between retrieval and generation layers.

**Hallucination mitigation via prompt design**
The system prompt explicitly constrains the LLM to answer only from retrieved context: *"If the answer is not in the context, say you don't have enough information."* Source chunks are surfaced to users alongside every answer for verification.

---

## Stack

| Layer | Technology |
|---|---|
| Frontend + backend | Streamlit |
| PDF parsing | PyMuPDF (fitz) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector store | ChromaDB (persistent) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq API (llama-3.1-8b-instant) |
| Deployment | Streamlit Cloud |

---

## Run locally

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com) (optional, for local LLM)

```bash
git clone https://github.com/anubhutipandey1/rag-chatbot
cd rag-chatbot
python -m venv venv
venv\Scripts\activate.bat       # Windows
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

Run:
```bash
streamlit run app.py
```

---

## Project structure

```
rag-chatbot/
├── app.py              # Main Streamlit app (ingestion + query pipeline)
├── evaluate.py         # RAG evaluation scripts
├── requirements.txt    # Dependencies
└── chroma_db/          # Local vector store (gitignored)
```

---

## What I'd add next

- **Hybrid search** — combine BM25 (keyword) with semantic search using Reciprocal Rank Fusion for better recall on exact-match queries
- **Evaluation pipeline** — Ragas metrics (faithfulness, answer relevancy, context recall) to measure retrieval quality systematically
- **Metadata filtering** — filter by document, date, or section before retrieval to reduce noise in multi-document scenarios
- **Streaming responses** — stream LLM output token-by-token for better perceived latency
