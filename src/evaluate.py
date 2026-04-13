import faiss, pickle, numpy as np, pandas as pd, requests
from sentence_transformers import SentenceTransformer
from pathlib import Path

VECTORSTORE_DIR = Path("data/vectorstore")

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

# filter keywords matched exactly to device names in our data
DEVICE_KEYWORDS = {
    "knee":        ["KNEE", "PROSTHESIS, KNEE", "TOTAL KNEE", "ATTUNE"],
    "pacemaker":   ["PACEMAKER", "ELECTRODE, PACEMAKER", "LEADLESS PACEMAKER",
                    "IMPLANTABLE PULSE GENERATOR", "PULSE GENERATOR, PACEMAKER"],
    "defibrillator": ["DEFIBRILLATOR", "CARDIOVERTER", "ICD"],
    "hip":         ["HIP", "PROSTHESIS, HIP", "PROSTHEISI, HIP", "PINNACLE",
                    "TRILOCK", "C-STEM"],
    "spinal":      ["SPINAL", "ORTHOSIS, SPINAL", "ORTHOSIS, CERVICAL",
                    "SPINAL CORD", "FIXATION, SPINAL"],
}

test_questions = [
    {"q": "What are the most common knee implant failure modes?",      "cat": "Knee",        "filter": "knee"},
    {"q": "What injuries have been caused by pacemaker leads?",        "cat": "Pacemaker",   "filter": "pacemaker"},
    {"q": "Which cardiac defibrillator models have the most reports?", "cat": "Cardiac",     "filter": "defibrillator"},
    {"q": "What patient outcomes follow hip implant loosening?",       "cat": "Hip",         "filter": "hip"},
    {"q": "How long after surgery do spinal implant failures occur?",  "cat": "Spinal",      "filter": "spinal"},
    {"q": "What fracture types are reported for knee tibial trays?",   "cat": "Knee",        "filter": "knee"},
    {"q": "Are there reports of pacemaker battery failure?",           "cat": "Pacemaker",   "filter": "pacemaker"},
    {"q": "What symptoms follow cardiac defibrillator malfunction?",   "cat": "Defibrillator","filter": "defibrillator"},
    {"q": "Which hip implant brands appear most in failure reports?",  "cat": "Hip",         "filter": "hip"},
    {"q": "What materials are linked to spinal implant corrosion?",    "cat": "Spinal",      "filter": "spinal"},
]

def matches_filter(device_name, filter_key):
    keywords = DEVICE_KEYWORDS.get(filter_key, [])
    device_upper = device_name.upper()
    return any(kw.upper() in device_upper for kw in keywords)

def ask(question, device_filter=None):
    vec = embedder.encode([question], convert_to_numpy=True)
    _, idxs = index.search(vec.astype(np.float32), k=100)

    if device_filter:
        filtered = [
            i for i in idxs[0]
            if matches_filter(metadata[i]["device_name"], device_filter)
        ]
        top_idxs = filtered[:4] if len(filtered) >= 2 else list(idxs[0][:4])
    else:
        top_idxs = list(idxs[0][:4])

    context = "\n\n".join([
        f"[Report: {metadata[i]['device_name']} | {metadata[i]['event_type']}]\n{chunks[i]}"
        for i in top_idxs
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
        return r.json().get("response",""), [metadata[i] for i in top_idxs]
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
    answer, sources = ask(item["q"], device_filter=item["filter"])
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
