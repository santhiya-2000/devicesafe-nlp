import faiss, pickle, numpy as np, pandas as pd, requests
from sentence_transformers import SentenceTransformer
from pathlib import Path

VECTORSTORE_DIR = Path("data/vectorstore")

# use enriched index if available, else base
if (VECTORSTORE_DIR / "maude_enriched.index").exists():
    index = faiss.read_index(str(VECTORSTORE_DIR / "maude_enriched.index"))
    with open(VECTORSTORE_DIR / "metadata_enriched.pkl", "rb") as f: metadata = pickle.load(f)
    with open(VECTORSTORE_DIR / "chunks_enriched.pkl",   "rb") as f: chunks   = pickle.load(f)
    print("Using enriched vectorstore")
else:
    index = faiss.read_index(str(VECTORSTORE_DIR / "maude.index"))
    with open(VECTORSTORE_DIR / "metadata.pkl", "rb") as f: metadata = pickle.load(f)
    with open(VECTORSTORE_DIR / "chunks.pkl",   "rb") as f: chunks   = pickle.load(f)
    print("Using base vectorstore")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

test_questions = [
    {"q": "What are the most common knee implant failure modes?",      "cat": "Knee"},
    {"q": "What injuries have been caused by pacemaker leads?",        "cat": "Pacemaker"},
    {"q": "Which cardiac defibrillator models have the most reports?", "cat": "Cardiac"},
    {"q": "What patient outcomes follow hip implant loosening?",       "cat": "Hip"},
    {"q": "How long after surgery do spinal implant failures occur?",  "cat": "Spinal"},
    {"q": "What fracture types are reported for knee tibial trays?",   "cat": "Knee"},
    {"q": "Are there reports of pacemaker battery failure?",           "cat": "Pacemaker"},
    {"q": "What symptoms follow cardiac defibrillator malfunction?",   "cat": "Cardiac"},
    {"q": "Which hip implant brands appear most in failure reports?",  "cat": "Hip"},
    {"q": "What materials are linked to spinal implant corrosion?",    "cat": "Spinal"},
]

def ask(question):
    vec = embedder.encode([question], convert_to_numpy=True)
    _, idxs = index.search(vec.astype(np.float32), k=4)
    context = "\n\n".join([
        f"[Report: {metadata[i]['device_name']} | {metadata[i]['event_type']}]\n{chunks[i]}"
        for i in idxs[0]
    ])
    prompt = f"""You are DeviceSafe NLP. Answer using ONLY the FDA reports below.
If the reports don't contain enough information, say so clearly.

FDA REPORTS:
{context}

QUESTION: {question}

ANSWER:"""
    try:
        r = requests.post("http://localhost:11434/api/generate",
            json={"model":"mistral","prompt":prompt,"stream":False}, timeout=180)
        return r.json().get("response",""), [metadata[i] for i in idxs[0]]
    except Exception as e:
        return f"Error: {e}", []

results = []
print("="*60)
print("DeviceSafe NLP — Chatbot Evaluation")
print("="*60)
print("Score each answer: y = pass, n = fail\n")

for i, item in enumerate(test_questions):
    print(f"\n[{i+1}/{len(test_questions)}] Category: {item['cat']}")
    print(f"Q: {item['q']}")
    answer, sources = ask(item["q"])
    print(f"\nA: {answer[:400]}...")
    print(f"Sources: {[s['device_name'] for s in sources[:2]]}")

    g = input("\nGrounded (y/n)? ").strip().lower() == "y"
    a = input("Accurate (y/n)? ").strip().lower() == "y"
    r = input("Relevant (y/n)? ").strip().lower() == "y"

    results.append({
        "num":      i+1,
        "category": item["cat"],
        "question": item["q"],
        "answer":   answer[:500],
        "sources":  " | ".join([s["device_name"] for s in sources]),
        "grounded": g,
        "accurate": a,
        "relevant": r,
        "all_pass": g and a and r
    })
    print(f"Scored: grounded={g} accurate={a} relevant={r}")

df = pd.DataFrame(results)
df.to_csv("data/processed/evaluation_results.csv", index=False)

print("\n" + "="*60)
print("FINAL EVALUATION SCORES")
print("="*60)
print(f"Grounded : {df.grounded.sum()}/{len(df)}  ({100*df.grounded.mean():.0f}%)")
print(f"Accurate : {df.accurate.sum()}/{len(df)}  ({100*df.accurate.mean():.0f}%)")
print(f"Relevant : {df.relevant.sum()}/{len(df)}  ({100*df.relevant.mean():.0f}%)")
print(f"All pass : {df.all_pass.sum()}/{len(df)}  ({100*df.all_pass.mean():.0f}%)")
print(f"\nSaved to data/processed/evaluation_results.csv")
