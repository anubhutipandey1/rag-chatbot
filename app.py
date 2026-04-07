import streamlit as st
import chromadb
from groq import Groq
import os
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()
import fitz
from sentence_transformers import SentenceTransformer, CrossEncoder

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("Document Q&A")
st.caption("Upload PDFs and ask questions across your documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

def get_collection_name(filename):
    return filename.replace(" ", "_").replace(".", "_").lower()

def extract_text(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def chunk_text(text, chunk_size=200, overlap=30):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def ingest_document(uploaded_file):
    filename = uploaded_file.name
    collection_name = get_collection_name(filename)

    with st.spinner(f"Reading {filename}..."):
        text = extract_text(uploaded_file)

    with st.spinner("Chunking text..."):
        chunks = chunk_text(text)

    with st.spinner("Embedding chunks..."):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(chunks, show_progress_bar=False)

    with st.spinner("Storing in ChromaDB..."):
        client = chromadb.PersistentClient(path="./chroma_db")
        try:
            client.delete_collection(collection_name)
        except:
            pass
        collection = client.create_collection(collection_name)
        collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            metadatas=[{"source": filename} for _ in chunks]
        )

    return collection_name, len(chunks)

def retrieve_and_rerank(query, selected_docs, top_k_retrieve=3, top_k_rerank=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])[0]

    client = chromadb.PersistentClient(path="./chroma_db")

    all_chunks = []
    all_sources = []

    for doc in selected_docs:
        collection_name = get_collection_name(doc)
        try:
            collection = client.get_collection(collection_name)
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k_retrieve, collection.count())
            )
            chunks = results["documents"][0]
            all_chunks.extend(chunks)
            all_sources.extend([doc] * len(chunks))
        except:
            pass

    if not all_chunks:
        return [], [], []

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, chunk] for chunk in all_chunks]
    scores = reranker.predict(pairs)

    scored = sorted(zip(all_chunks, all_sources, scores), key=lambda x: x[2], reverse=True)
    top_chunks = [chunk for chunk, source, score in scored[:top_k_rerank]]
    top_sources = [source for chunk, source, score in scored[:top_k_rerank]]

    return top_chunks, top_sources, [s for _, _, s in scored[:top_k_rerank]]

def generate_answer(query, chunks, sources):
    context_parts = []
    for chunk, source in zip(chunks, sources):
        context_parts.append(f"[From: {source}]\n{chunk}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information in the selected documents to answer this."
Do not make up information.

Context:
{context}

Question: {query}

Answer:"""

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

with st.sidebar:
    st.header("Documents")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file is not None:
        already_uploaded = any(d["name"] == uploaded_file.name for d in st.session_state.uploaded_docs)
        if not already_uploaded:
            if st.button("Process Document", type="primary"):
                collection_name, num_chunks = ingest_document(uploaded_file)
                st.session_state.uploaded_docs.append({
                    "name": uploaded_file.name,
                    "collection": collection_name,
                    "chunks": num_chunks
                })
                st.success(f"Done! {num_chunks} chunks stored.")
        else:
            st.info("This document is already uploaded.")

    if not st.session_state.uploaded_docs:
        st.divider()
        st.caption("No PDF? Try the sample document:")
        if st.button("Load sample: insomnia.pdf", type="secondary"):
            sample_path = os.path.join(os.path.dirname(__file__), "insomnia.pdf")
            with open(sample_path, "rb") as f:
                sample_file = BytesIO(f.read())
                sample_file.name = "insomnia.pdf"
                collection_name, num_chunks = ingest_document(sample_file)
                st.session_state.uploaded_docs.append({
                    "name": "insomnia.pdf",
                    "collection": collection_name,
                    "chunks": num_chunks
                })
                st.rerun()

    if st.session_state.uploaded_docs:
        st.divider()
        st.subheader("Select documents to search")
        selected_docs = []
        for doc in st.session_state.uploaded_docs:
            checked = st.checkbox(
                f"{doc['name']} ({doc['chunks']} chunks)",
                value=True,
                key=f"doc_{doc['name']}"
            )
            if checked:
                selected_docs.append(doc["name"])
    else:
        selected_docs = []
        st.info("No documents uploaded yet.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message:
            with st.expander("Sources"):
                for i, (chunk, source) in enumerate(zip(message["chunks"], message["sources"])):
                    st.caption(f"Chunk {i+1} — from **{source}**")
                    st.caption(chunk[:200] + "...")

if prompt := st.chat_input("Ask a question about your selected documents..."):
    if not selected_docs:
        st.warning("Please upload a document and select it from the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chunks, sources, scores = retrieve_and_rerank(prompt, selected_docs)
                answer = generate_answer(prompt, chunks, sources)
            st.write(answer)
            with st.expander("Sources"):
                for i, (chunk, source, score) in enumerate(zip(chunks, sources, scores)):
                    if score > 5:
                        relevance = "High"
                    elif score > 0:
                        relevance = "Medium"
                    else:
                        relevance = "Low"
                    st.caption(f"Chunk {i+1} — from **{source}** | Relevance: {relevance}")
                    st.caption(chunk[:200] + "...")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "chunks": chunks,
            "sources": sources
        })
