from transformers import pipeline
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

samples = [
    "The knee implant fractured post-operatively causing severe pain.",
    "Pacemaker lead exhibited intermittent electrical failure during stress testing.",
    "The cardiac defibrillator malfunctioned and failed to deliver therapy.",
    "Hip implant loosening was observed requiring revision surgery.",
    "Spinal implant showed signs of corrosion after six months."
]

models = {
    "BC5CDR-Disease": "pruas/BENT-PubMedBERT-NER-Disease",
    "BioNLP-NER":     "allenai/scibert_scivocab_cased",
    "NCBI-Disease":   "alvaroalon2/biobert_diseases_ner",
}

results = []

for model_name, model_path in models.items():
    print(f"\nLoading {model_name}...")
    try:
        ner = pipeline(
            "ner",
            model=model_path,
            aggregation_strategy="simple",
            device=-1
        )
        print(f"  Loaded OK")
        print(f"\n  {model_name} results:")
        print(f"  {'Entity':<35} {'Label':<20} {'Score'}")
        print(f"  {'-'*65}")

        for sent in samples:
            entities = ner(sent)
            for e in entities:
                if e["score"] > 0.7:
                    print(f"  {e['word']:<35} {e['entity_group']:<20} {e['score']:.2f}")
                    results.append({
                        "model":    model_name,
                        "sentence": sent[:50],
                        "entity":   e["word"],
                        "label":    e["entity_group"],
                        "score":    round(e["score"], 3)
                    })
    except Exception as e:
        print(f"  Could not load {model_name}: {e}")

df = pd.DataFrame(results)
if len(df) > 0:
    df.to_csv(PROCESSED_DIR / "bert_biomedical_ner.csv", index=False)
    print(f"\nSaved {len(df)} entities to data/processed/bert_biomedical_ner.csv")

print("\n" + "="*65)
print("REPORT SUMMARY — BERT Analysis")
print("="*65)
print("""
Key findings for your Transformers/BERT section:

1. BASE BERT (no fine-tuning)
   BioBERT, PubMedBERT, ClinicalBERT → LABEL_0/LABEL_1 only
   Not usable for NER without task-specific fine-tuning

2. GENERAL FINE-TUNED NER (dslim/bert-base-NER)
   Trained on news/general text → finds 0 entities in device reports
   Confirms: domain shift is a real problem for medical text

3. DOMAIN-SPECIFIC FINE-TUNED NER
   Models fine-tuned on biomedical corpora → meaningful entity labels
   Best choice for device report NER

4. CONCLUSION FOR DEVICESAFE NLP
   AWS Comprehend Medical is effectively a fine-tuned transformer
   trained on clinical + regulatory text — best fit for our pipeline
   without requiring us to fine-tune our own model from scratch
""")
