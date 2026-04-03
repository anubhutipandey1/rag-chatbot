import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama

def retrieve_chunks(query, top_k=5):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])[0]
    
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("documents")
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    
    return results["documents"][0]

def rerank_chunks(query, chunks, top_k=3):
    print("Reranking chunks...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    pairs = [[query, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    
    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    print("\nReranker scores:")
    for i, (chunk, score) in enumerate(scored_chunks):
        print(f"Chunk {i+1} (score: {round(float(score), 4)}): {chunk[:100]}...")
    
    top_chunks = [chunk for chunk, score in scored_chunks[:top_k]]
    return top_chunks

def generate_answer(query, chunks):
    context = "\n\n---\n\n".join(chunks)
    
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information in the document to answer this."
Do not make up information.

Context:
{context}

Question: {query}

Answer:"""
    
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response["message"]["content"]

def rag_with_reranking(query):
    print(f"\nQuestion: {query}")
    print("-" * 50)
    
    print(f"Step 1: Retrieving top 5 chunks from ChromaDB...")
    chunks = retrieve_chunks(query, top_k=5)
    print(f"Got {len(chunks)} chunks")
    
    print("\nStep 2: Reranking...")
    reranked_chunks = rerank_chunks(query, chunks, top_k=3)
    
    print("\nStep 3: Generating answer from top 3 reranked chunks...")
    answer = generate_answer(query, reranked_chunks)
    
    print(f"\nFinal Answer: {answer}")
    return answer

rag_with_reranking("what are the 3 PM interview question types?")