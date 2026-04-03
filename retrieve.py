import chromadb
from sentence_transformers import SentenceTransformer

def retrieve(query, top_k=3):
    print(f"\nQuery: {query}")
    
    print("Embedding query...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])[0]
    
    print("Searching ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("documents")
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    
    print(f"\nTop {top_k} chunks found:\n")
    chunks = results["documents"][0]
    distances = results["distances"][0]
    
    for i, (chunk, distance) in enumerate(zip(chunks, distances)):
        print(f"--- Chunk {i+1} (distance: {round(distance, 4)}) ---")
        print(chunk[:300])
        print()
    
    return chunks

retrieve("what are the 3 PM interview question types?")