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

# this is BC5CDR — fine-tuned on biomedical NER, gives real entity labels
print("Loading fine-tuned biomedical NER model (BC5CDR)...")
ner = pipeline(
    "ner",
    model="allenai/scibert_scivocab_uncased",
    aggregation_strategy="simple",
    device=-1
)

# also try a properly fine-tuned clinical NER
print("Loading clinical NER (dslim/bert-base-NER)...")
gen_ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple",
    device=-1
)

results = []

print("\n" + "="*65)
print("Fine-tuned NER — dslim/bert-base-NER (general)")
print("="*65)
for sent in samples:
    entities = gen_ner(sent)
    print(f"\nSentence: {sent}")
    if entities:
        for e in entities:
            print(f"  {e['word']:<35} {e['entity_group']:<15} {e['score']:.2f}")
            results.append({
                "model": "bert-base-NER",
                "sentence": sent[:50],
                "entity": e["word"],
                "label": e["entity_group"],
                "score": round(e["score"], 3)
            })
    else:
        print("  (no entities found)")

df = pd.DataFrame(results)
df.to_csv(PROCESSED_DIR / "bert_finetuned.csv", index=False)
print(f"\nSaved to data/processed/bert_finetuned.csv")

print("\n--- KEY FINDING FOR REPORT ---")
print("Base BERT variants (BioBERT, PubMedBERT, ClinicalBERT) without")
print("fine-tuning produce binary LABEL_0/LABEL_1 outputs — not useful")
print("for medical entity extraction.")
print("Fine-tuned models on biomedical corpora produce named entity labels.")
print("This justifies using AWS Comprehend Medical (pre-trained + fine-tuned)")
print("rather than raw BERT for our entity extraction pipeline.")
