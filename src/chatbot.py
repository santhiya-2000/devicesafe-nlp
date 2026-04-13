import settings  # noqa: I001 — set OMP/thread env before numpy/torch

import pickle
from pathlib import Path

import faiss
import requests

from retrieve_utils import retrieve_rag

VECTORSTORE_DIR = Path("data/vectorstore")

print("Loading vectorstore...")
index    = faiss.read_index(str(VECTORSTORE_DIR / "maude.index"))
with open(VECTORSTORE_DIR / "metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
with open(VECTORSTORE_DIR / "chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("Loading embedding model...")
embedder = settings.load_sentence_transformer()

print("Vectorstore ready.\n")

def ask_ollama(prompt):
    try:
        r = requests.post(
            settings.OLLAMA_URL,
            json=settings.ollama_generate_json(prompt),
            timeout=300,
        )
        return r.json().get("response", "No response from Ollama")
    except Exception as e:
        return f"Ollama error: {e}. Use a smaller model: DEVICESAFE_OLLAMA_MODEL=phi3:mini"

def chat(question):
    print(f"\nQuestion: {question}")
    print("Retrieving relevant reports...")

    results, note = retrieve_rag(question, index, metadata, chunks, embedder, k=4)
    if note:
        print(f"Note: {note}\n")

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
