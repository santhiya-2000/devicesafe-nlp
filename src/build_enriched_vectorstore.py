import settings  # noqa: I001 — set OMP/thread env before numpy/torch

import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

PROCESSED_DIR   = Path("data/processed")
VECTORSTORE_DIR = Path("data/vectorstore")
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROCESSED_DIR / "maude_processed.csv")
df["clean_text"] = df["clean_text"].fillna("")
df = df[df["clean_text"].str.len() > 50].reset_index(drop=True)

entity_df = pd.read_csv(PROCESSED_DIR / "entity_summary.csv")
entity_df["report_id"] = entity_df["report_id"].astype(str)
df["report_id"] = df["report_id"].astype(str)
df = df.merge(entity_df[["report_id","conditions","anatomy","procedures"]],
              on="report_id", how="left")
print(f"Loaded {len(df)} reports with entity data merged")

model = settings.load_sentence_transformer()

chunks   = []
metadata = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building chunks"):
    text  = str(row["clean_text"])
    words = text.split()

    conditions = str(row.get("conditions", "")) if pd.notna(row.get("conditions")) else ""
    anatomy    = str(row.get("anatomy",    "")) if pd.notna(row.get("anatomy"))    else ""
    procedures = str(row.get("procedures", "")) if pd.notna(row.get("procedures")) else ""

    entity_context = ""
    if conditions: entity_context += f" [conditions: {conditions}]"
    if anatomy:    entity_context += f" [anatomy: {anatomy}]"
    if procedures: entity_context += f" [procedures: {procedures}]"

    for i in range(0, len(words), 80):
        chunk = " ".join(words[i:i+80])
        if len(chunk) > 50:
            enriched = chunk + entity_context if i == 0 else chunk
            chunks.append(enriched)
            metadata.append({
                "report_id":   str(row.get("report_id", idx)),
                "device_name": str(row.get("device_name", "")),
                "event_type":  str(row.get("event_type", "")),
                "date":        str(row.get("date", "")),
                "chunk_text":  chunk,
                "conditions":  conditions,
                "anatomy":     anatomy,
                "procedures":  procedures,
            })

print(f"Created {len(chunks)} enriched chunks")

print("Generating embeddings...")
embeddings = model.encode(
    chunks,
    batch_size=settings.ENCODE_BATCH,
    show_progress_bar=True,
    convert_to_numpy=True,
)

print("Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype(np.float32))

faiss.write_index(index, str(VECTORSTORE_DIR / "maude_enriched.index"))
with open(VECTORSTORE_DIR / "metadata_enriched.pkl", "wb") as f:
    pickle.dump(metadata, f)
with open(VECTORSTORE_DIR / "chunks_enriched.pkl", "wb") as f:
    pickle.dump(chunks, f)

print(f"\nEnriched vectorstore saved — {index.ntotal} vectors")

print("\nTesting retrieval...")
query = "knee implant fracture causing pain"
q_vec = model.encode([query], convert_to_numpy=True)
distances, indices = index.search(q_vec.astype(np.float32), k=3)
print(f"\nTop 3 for: '{query}'")
for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    m = metadata[idx]
    print(f"  Rank {rank+1} | {m['device_name']} | {m['event_type']}")
    print(f"  Conditions : {m['conditions'][:80]}")
    print(f"  Text       : {m['chunk_text'][:120]}...")
