import chromadb
import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder
from test_questions import test_cases

def retrieve_and_rerank(query, top_k_retrieve=5, top_k_rerank=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])[0]
    
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("documents")
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k_retrieve
    )
    chunks = results["documents"][0]
    
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    
    scored_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scored_chunks[:top_k_rerank]]

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

def evaluate_faithfulness(answer, chunks):
    context = "\n\n".join(chunks)
    
    prompt = f"""You are an evaluator. Given the context and an answer, determine if the answer is fully supported by the context.

Context:
{context}

Answer:
{answer}

Is every claim in the answer supported by the context? 
Respond with only a number between 0 and 1 where:
1.0 = fully supported
0.5 = partially supported  
0.0 = not supported or contradicts context

Score:"""
    
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    
    try:
        score = float(response["message"]["content"].strip().split()[0])
        return round(min(max(score, 0.0), 1.0), 2)
    except:
        return 0.5

def evaluate_relevance(question, answer):
    prompt = f"""You are an evaluator. Determine if the answer actually addresses the question asked.

Question: {question}

Answer: {answer}

Does the answer directly address the question?
Respond with only a number between 0 and 1 where:
1.0 = directly and completely answers the question
0.5 = partially answers the question
0.0 = does not answer the question at all

Score:"""
    
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    
    try:
        score = float(response["message"]["content"].strip().split()[0])
        return round(min(max(score, 0.0), 1.0), 2)
    except:
        return 0.5

def run_evaluation():
    print("=" * 60)
    print("RAG SYSTEM EVALUATION")
    print("=" * 60)
    
    total_faithfulness = 0
    total_relevance = 0
    
    for i, test_case in enumerate(test_cases):
        question = test_case["question"]
        expected = test_case["expected"]
        
        print(f"\nTest {i+1}: {question}")
        print(f"Expected: {expected}")
        print("-" * 40)
        
        chunks = retrieve_and_rerank(question)
        answer = generate_answer(question, chunks)
        
        print(f"Answer: {answer}")
        
        faithfulness = evaluate_faithfulness(answer, chunks)
        relevance = evaluate_relevance(question, answer)
        
        total_faithfulness += faithfulness
        total_relevance += relevance
        
        print(f"Faithfulness score:     {faithfulness}")
        print(f"Answer relevance score: {relevance}")
    
    avg_faithfulness = round(total_faithfulness / len(test_cases), 2)
    avg_relevance = round(total_relevance / len(test_cases), 2)
    overall = round((avg_faithfulness + avg_relevance) / 2, 2)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Average faithfulness:     {avg_faithfulness}")
    print(f"Average answer relevance: {avg_relevance}")
    print(f"Overall score:            {overall}")
    print("=" * 60)
    
    if avg_faithfulness < 0.7:
        print("⚠ Faithfulness is low — consider tightening your prompt template")
    if avg_relevance < 0.7:
        print("⚠ Relevance is low — consider improving your chunking strategy")
    if overall >= 0.8:
        print("✓ System is performing well overall")

run_evaluation()