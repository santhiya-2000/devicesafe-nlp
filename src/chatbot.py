import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import requests
import json

VECTORSTORE_DIR = Path("data/vectorstore")

print("Loading vectorstore...")
index    = faiss.read_index(str(VECTORSTORE_DIR / "maude.index"))
with open(VECTORSTORE_DIR / "metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
with open(VECTORSTORE_DIR / "chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Vectorstore ready.\n")

def retrieve(query, k=4):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec.astype(np.float32), k=k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "text":        chunks[idx],
            "device":      metadata[idx]["device_name"],
            "event_type":  metadata[idx]["event_type"],
            "report_id":   metadata[idx]["report_id"],
            "distance":    round(float(dist), 3)
        })
    return results

def ask_ollama(prompt, model="mistral"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        r = requests.post(url, json=payload, timeout=180)
        return r.json().get("response", "No response from Ollama")
    except Exception as e:
        return f"Ollama error: {e}. Make sure Ollama is running with: ollama serve"

def chat(question):
    print(f"\nQuestion: {question}")
    print("Retrieving relevant reports...")

    results = retrieve(question, k=4)

    context = ""
    for i, r in enumerate(results):
        context += f"\n[Report {i+1} | Device: {r['device']} | Event: {r['event_type']}]\n{r['text']}\n"

    prompt = f"""You are DeviceSafe NLP, an AI assistant that answers questions about medical device failures using real FDA adverse event reports.

Use ONLY the following FDA report excerpts to answer the question. Do not make up information.
If the reports do not contain enough information, say so clearly.

FDA REPORT EXCERPTS:
{context}

QUESTION: {question}

ANSWER (based only on the FDA reports above):"""

    print("Generating answer...")
    answer = ask_ollama(prompt)

    print("\n" + "="*60)
    print("ANSWER:")
    print("="*60)
    print(answer)
    print("\nSOURCES:")
    for i, r in enumerate(results):
        print(f"  [{i+1}] Device: {r['device']} | Event: {r['event_type']} | Report: {r['report_id']}")

    return answer

if __name__ == "__main__":
    print("="*60)
    print("DeviceSafe NLP — RAG Chatbot")
    print("Type 'quit' to exit")
    print("="*60)

    test_questions = [
        "What are the most common knee implant failure modes?",
        "What pacemaker failures have been reported?",
        "What injuries are associated with hip implants?",
    ]

    print("\nRunning test questions first...\n")
    for q in test_questions:
        chat(q)
        print("\n" + "-"*60)

    print("\nNow entering interactive mode...")
    while True:
        question = input("\nAsk a question: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            break
        if question:
            chat(question)
