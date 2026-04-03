import google.generativeai as genai
import chromadb
import os
from sentence_transformers import SentenceTransformer

def retrieve_chunks(query, top_k=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])[0]
    
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("documents")
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    
    return results["documents"][0]

import ollama
import chromadb
from sentence_transformers import SentenceTransformer

def retrieve_chunks(query, top_k=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])[0]
    
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("documents")
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    
    return results["documents"][0]

def generate_answer(query):
    print(f"\nQuestion: {query}")
    
    print("Retrieving relevant chunks...")
    chunks = retrieve_chunks(query)
    
    context = "\n\n---\n\n".join(chunks)
    
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information in the document to answer this."
Do not make up information.

Context:
{context}

Question: {query}

Answer:"""
    
    print("Asking Llama...\n")
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer = response["message"]["content"]
    print(f"Answer: {answer}")
    
    print("\n--- Sources used ---")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk[:150]}...")
    
    return answer

generate_answer("what is the capital of France?")