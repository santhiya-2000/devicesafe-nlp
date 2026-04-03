from transformers import pipeline, AutoTokenizer, AutoModel
import pandas as pd
import torch
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

# sample device report sentences for comparison
samples = [
    "The knee implant fractured post-operatively causing severe pain.",
    "Pacemaker lead exhibited intermittent electrical failure during stress testing.",
    "The cardiac defibrillator malfunctioned and failed to deliver therapy.",
    "Hip implant loosening was observed requiring revision surgery.",
    "Spinal implant showed signs of corrosion after six months."
]

models = {
    "BioBERT":      "dmis-lab/biobert-base-cased-v1.2",
    "PubMedBERT":   "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
}

print("DeviceSafe NLP — BERT Variant Comparison")
print("="*60)
print("Running NER on device failure report samples...\n")

results = []

for model_name, model_path in models.items():
    print(f"Loading {model_name}...")
    try:
        ner = pipeline(
            "ner",
            model=model_path,
            aggregation_strategy="simple",
            device=-1  # CPU
        )
        print(f"  {model_name} loaded OK")

        for sent in samples:
            entities = ner(sent)
            for ent in entities:
                results.append({
                    "model":      model_name,
                    "sentence":   sent[:60] + "...",
                    "entity":     ent["word"],
                    "label":      ent["entity_group"],
                    "score":      round(ent["score"], 3)
                })
    except Exception as e:
        print(f"  Error loading {model_name}: {e}")

df = pd.DataFrame(results)

if len(df) > 0:
    print("\n" + "="*60)
    print("ENTITIES EXTRACTED PER MODEL")
    print("="*60)
    for model_name in models.keys():
        subset = df[df["model"] == model_name]
        print(f"\n{model_name} — {len(subset)} entities found:")
        print(subset[["entity","label","score"]].to_string(index=False))

    print("\n" + "="*60)
    print("SUMMARY — entities found per model")
    print("="*60)
    print(df.groupby("model")["entity"].count().to_string())

    df.to_csv(PROCESSED_DIR / "bert_comparison.csv", index=False)
    print(f"\nSaved to data/processed/bert_comparison.csv")
else:
    print("No results — check model loading errors above")
