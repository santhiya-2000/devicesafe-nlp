import pandas as pd
import spacy
import re
from pathlib import Path
from tqdm import tqdm

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("Loading scispaCy model...")
nlp = spacy.load("en_core_sci_md")
print("Model loaded.")

df = pd.read_csv(RAW_DIR / "maude_reports.csv")
print(f"Loaded {len(df)} reports")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # remove redacted markers like (B)(4), (B)(6)
    text = re.sub(r'\(B\)\(\d+\)', '', text)
    # remove special chars like ¿
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # lowercase
    text = text.lower()
    return text

def process_with_scispacy(text):
    if not text:
        return [], [], []
    doc = nlp(text)
    tokens    = [token.text   for token in doc if not token.is_space]
    pos_tags  = [token.pos_   for token in doc if not token.is_space]
    lemmas    = [token.lemma_ for token in doc if not token.is_space]
    return tokens, pos_tags, lemmas

print("\nCleaning and processing reports...")
tqdm.pandas()

df["clean_text"] = df["narrative_text"].progress_apply(clean_text)
df = df[df["clean_text"].str.len() > 50].reset_index(drop=True)
print(f"After cleaning: {len(df)} reports")

print("\nRunning scispaCy POS tagging (this takes ~2 mins)...")
results = []
for text in tqdm(df["clean_text"]):
    tokens, pos_tags, lemmas = process_with_scispacy(text)
    results.append({
        "tokens":   " ".join(tokens),
        "pos_tags": " ".join(pos_tags),
        "lemmas":   " ".join(lemmas)
    })

results_df = pd.DataFrame(results)
df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

df.to_csv(PROCESSED_DIR / "maude_processed.csv", index=False)
print(f"\nSaved {len(df)} processed reports to data/processed/maude_processed.csv")

print("\n--- Sample output ---")
sample = df.iloc[1]
print(f"Original:  {sample['narrative_text'][:150]}...")
print(f"Cleaned:   {sample['clean_text'][:150]}...")
print(f"Tokens:    {sample['tokens'][:150]}...")
print(f"POS tags:  {sample['pos_tags'][:150]}...")
print(f"Lemmas:    {sample['lemmas'][:150]}...")
