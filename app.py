import streamlit as st
import chromadb
from groq import Groq
import os
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()
import fitz
from sentence_transformers import SentenceTransformer, CrossEncoder
 
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return embedder, reranker
 
@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path="./chroma_db")
 
st.set_page_config(page_title="RAG Document Q&A", layout="wide", page_icon="📄")
 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
 
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
 
    .stApp {
        background-color: #f8f9fb;
    }
 
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 860px;
    }
 
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e8eaed;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }
 
    [data-testid="stChatMessage"] {
        background-color: #ffffff;
        border: 1px solid #e8eaed;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
 
    .stButton > button {
        border-radius: 8px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.15s ease;
    }
    .stButton > button[kind="primary"] {
        background: #4f7cff;
        border: none;
        color: white;
    }
    .stButton > button[kind="primary"]:hover {
        background: #3a66e8;
        box-shadow: 0 4px 12px rgba(79,124,255,0.3);
        transform: translateY(-1px);
    }
    .stButton > button[kind="secondary"] {
        background: #f0f4ff;
        border: 1px solid #c7d4ff;
        color: #4f7cff;
    }
 
    [data-testid="stExpander"] {
        border: 1px solid #e8eaed;
        border-radius: 8px;
        background: #ffffff;
    }
 
    [data-testid="stFileUploader"] {
        background: #f8f9fb;
        border: 1.5px dashed #c7d4ff;
        border-radius: 10px;
    }
 
    hr {
        border-color: #e8eaed;
    }
 
    .tech-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 500;
        margin-right: 6px;
        margin-bottom: 4px;
        font-family: 'DM Mono', monospace;
    }
 
    .pipeline-step {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        padding: 10px 0;
        border-bottom: 1px solid #f0f0f0;
    }
    .pipeline-step:last-child {
        border-bottom: none;
    }
    .step-num {
        background: #4f7cff;
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 600;
        flex-shrink: 0;
    }
</style>
""", unsafe_allow_html=True)
 
 
# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px;">
        <div style="background: linear-gradient(135deg, #4f7cff, #7c3aed); width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem;">📄</div>
        <div>
            <h1 style="margin: 0; font-size: 1.6rem; font-weight: 600; color: #111827; font-family: 'DM Sans', sans-serif;">RAG Document Q&A</h1>
            <p style="margin: 0; font-size: 0.875rem; color: #6b7280;">Ask questions across your documents — answers grounded in your content, never made up.</p>
        </div>
    </div>
    <div style="margin-top: 10px;">
        <span class="tech-badge" style="background: #eff6ff; color: #1d4ed8;">🔍 Semantic Search</span>
        <span class="tech-badge" style="background: #f0fdf4; color: #15803d;">⚡ Cross-Encoder Reranking</span>
        <span class="tech-badge" style="background: #fff7ed; color: #c2410c;">🦙 Llama 3.1 · Groq</span>
        <span class="tech-badge" style="background: #faf5ff; color: #7c3aed;">🗄️ ChromaDB</span>
    </div>
</div>
<hr style="margin-bottom: 1.5rem;">
""", unsafe_allow_html=True)
 
 
# ── Quick Guide ───────────────────────────────────────────────────────────────
with st.expander("📖  How to use this app", expanded=False):
    st.markdown("""
    <div style="font-family: 'DM Sans', sans-serif; padding: 0.5rem 0;">
 
    <div class="pipeline-step">
        <div class="step-num">1</div>
        <div>
            <strong style="color: #111827;">Upload a PDF</strong><br>
            <span style="color: #6b7280; font-size: 0.875rem;">Use the sidebar on the left to upload one or more PDF documents. Click <strong>Process Document</strong> to embed and index it.</span>
        </div>
    </div>
 
    <div class="pipeline-step">
        <div class="step-num">2</div>
        <div>
            <strong style="color: #111827;">Select documents to search</strong><br>
            <span style="color: #6b7280; font-size: 0.875rem;">Once uploaded, check the documents you want to query. You can search across multiple documents at once.</span>
        </div>
    </div>
 
    <div class="pipeline-step">
        <div class="step-num">3</div>
        <div>
            <strong style="color: #111827;">Ask a question</strong><br>
            <span style="color: #6b7280; font-size: 0.875rem;">Type your question in the chat box below. The app retrieves the most relevant chunks, reranks them, and generates a grounded answer.</span>
        </div>
    </div>
 
    <div class="pipeline-step">
        <div class="step-num">4</div>
        <div>
            <strong style="color: #111827;">Check your sources</strong><br>
            <span style="color: #6b7280; font-size: 0.875rem;">Every answer includes a <strong>Sources</strong> expander showing the exact document chunks used — so you can verify the answer yourself.</span>
        </div>
    </div>
 
    <div style="margin-top: 1rem; padding: 10px 14px; background: #f0f4ff; border-left: 3px solid #4f7cff; border-radius: 6px;">
        <span style="font-size: 0.825rem; color: #374151;">💡 <strong>Tip:</strong> If the answer isn't in your documents, the app will tell you — it won't make things up.</span>
    </div>
 
    </div>
    """, unsafe_allow_html=True)
 
 
# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
 
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []
 
 
# ── Helper functions ──────────────────────────────────────────────────────────
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
        embedder, _ = load_models()
        embeddings = embedder.encode(chunks, show_progress_bar=False)
 
    with st.spinner("Storing in ChromaDB..."):
        client = get_chroma_client()
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
    embedder, reranker = load_models()
    query_embedding = embedder.encode([query])[0]
 
    client = get_chroma_client()
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
    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
 
    return stream
 
 
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="font-family: 'DM Sans', sans-serif; font-weight: 600; color: #111827; margin-bottom: 2px;">Documents</h3>
        <p style="font-size: 0.8rem; color: #9ca3af; margin: 0;">Upload PDFs to query</p>
    </div>
    """, unsafe_allow_html=True)
 
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", label_visibility="collapsed")
 
    if uploaded_file is not None:
        already_uploaded = any(d["name"] == uploaded_file.name for d in st.session_state.uploaded_docs)
        if not already_uploaded:
            if st.button("⚙️  Process Document", type="primary", use_container_width=True):
                collection_name, num_chunks = ingest_document(uploaded_file)
                st.session_state.uploaded_docs.append({
                    "name": uploaded_file.name,
                    "collection": collection_name,
                    "chunks": num_chunks
                })
                st.success(f"✅ Done — {num_chunks} chunks indexed.")
        else:
            st.info("Already uploaded.")
 
    if not st.session_state.uploaded_docs:
        st.divider()
        st.markdown("<p style='font-size: 0.8rem; color: #9ca3af;'>No PDF? Try the sample:</p>", unsafe_allow_html=True)
        if st.button("📄  Load sample: insomnia.pdf", type="secondary", use_container_width=True):
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
        st.markdown("<p style='font-size: 0.8rem; font-weight: 600; color: #374151; margin-bottom: 6px;'>Select documents to search</p>", unsafe_allow_html=True)
        selected_docs = []
        for doc in st.session_state.uploaded_docs:
            checked = st.checkbox(
                f"{doc['name']}",
                value=True,
                key=f"doc_{doc['name']}",
                help=f"{doc['chunks']} chunks indexed"
            )
            if checked:
                selected_docs.append(doc["name"])
 
        st.divider()
        if st.button("🗑️  Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    else:
        selected_docs = []
        st.markdown("""
        <div style="margin-top: 1rem; padding: 12px; background: #f8f9fb; border-radius: 8px; text-align: center;">
            <p style="font-size: 0.8rem; color: #9ca3af; margin: 0;">Upload a PDF above to get started</p>
        </div>
        """, unsafe_allow_html=True)
 
    st.divider()
    st.markdown("""
    <div style="font-size: 0.72rem; color: #9ca3af; line-height: 1.6;">
        Built with Streamlit · ChromaDB · Groq<br>
        <a href="https://github.com/anubhutipandey1/rag-chatbot" target="_blank" style="color: #4f7cff; text-decoration: none;">View on GitHub →</a>
    </div>
    """, unsafe_allow_html=True)
 
 
# ── Welcome state ─────────────────────────────────────────────────────────────
if not st.session_state.uploaded_docs and not st.session_state.messages:
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; color: #9ca3af;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📂</div>
        <p style="font-size: 1rem; font-weight: 500; color: #6b7280; margin-bottom: 6px;">No documents uploaded yet</p>
        <p style="font-size: 0.875rem;">Upload a PDF in the sidebar to start asking questions.</p>
    </div>
    """, unsafe_allow_html=True)
 
 
# ── Chat history ──────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message:
            with st.expander("📎  Sources"):
                for i, (chunk, source) in enumerate(zip(message["chunks"], message["sources"])):
                    st.markdown(f"<span style='font-size: 0.75rem; font-weight: 600; color: #4f7cff;'>Chunk {i+1} · {source}</span>", unsafe_allow_html=True)
                    st.caption(chunk[:200] + "...")
 
 
# ── Chat input ────────────────────────────────────────────────────────────────
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
                stream = generate_answer(prompt, chunks, sources)
 
            def stream_tokens():
                for chunk in stream:
                    token = chunk.choices[0].delta.content
                    if token:
                        yield token
 
            answer = st.write_stream(stream_tokens())
 
            with st.expander("📎  Sources"):
                for i, (chunk, source, score) in enumerate(zip(chunks, sources, scores)):
                    if score > 5:
                        relevance = "🟢 High"
                        rel_color = "#15803d"
                    elif score > 0:
                        relevance = "🟡 Medium"
                        rel_color = "#b45309"
                    else:
                        relevance = "🔴 Low"
                        rel_color = "#b91c1c"
                    st.markdown(
                        f"<span style='font-size: 0.75rem; font-weight: 600; color: #4f7cff;'>Chunk {i+1} · {source}</span>"
                        f"<span style='font-size: 0.72rem; color: {rel_color}; margin-left: 8px;'>{relevance}</span>",
                        unsafe_allow_html=True
                    )
                    st.caption(chunk[:200] + "...")
 
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "chunks": chunks,
            "sources": sources
        })
 