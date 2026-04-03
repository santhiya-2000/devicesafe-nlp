import pandas as pd
import faiss
import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROCESSED_DIR = Path("data/processed")
VECTORSTORE_DIR = Path("data/vectorstore")
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

print("Loading processed reports...")
df = pd.read_csv(PROCESSED_DIR / "maude_processed.csv")
df["clean_text"] = df["clean_text"].fillna("")
df = df[df["clean_text"].str.len() > 50].reset_index(drop=True)
print(f"Loaded {len(df)} reports")

print("\nLoading sentence transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded OK")

print("\nBuilding text chunks for embedding...")
chunks = []
metadata = []

for idx, row in df.iterrows():
    text = row["clean_text"]
    # split long reports into chunks of ~500 chars
    words = text.split()
    chunk_size = 80  # ~80 words per chunk
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        if len(chunk) > 50:
            chunks.append(chunk)
            metadata.append({
                "report_id":   str(row.get("report_id", idx)),
                "device_name": str(row.get("device_name", "")),
                "event_type":  str(row.get("event_type", "")),
                "date":        str(row.get("date", "")),
                "chunk_text":  chunk,
                "chunk_idx":   i // chunk_size
            })

print(f"Created {len(chunks)} text chunks from {len(df)} reports")

print("\nGenerating embeddings (this takes ~2 mins)...")
embeddings = model.encode(
    chunks,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)
print(f"Embeddings shape: {embeddings.shape}")

print("\nBuilding FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype(np.float32))
print(f"FAISS index built — {index.ntotal} vectors, {dimension} dimensions")

print("\nSaving vectorstore...")
faiss.write_index(index, str(VECTORSTORE_DIR / "maude.index"))
with open(VECTORSTORE_DIR / "metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)
with open(VECTORSTORE_DIR / "chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Saved to data/vectorstore/")

print("\nTesting retrieval with sample query...")
query = "knee implant fracture causing pain"
query_embedding = model.encode([query], convert_to_numpy=True)
distances, indices = index.search(query_embedding.astype(np.float32), k=3)

print(f"\nTop 3 results for: '{query}'")
print("="*60)
for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    meta = metadata[idx]
    print(f"\nRank {rank+1} (distance: {dist:.2f})")
    print(f"  Device : {meta['device_name']}")
    print(f"  Event  : {meta['event_type']}")
    print(f"  Text   : {meta['chunk_text'][:150]}...")

print("\nVectorstore build complete.")
