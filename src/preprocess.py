import settings  # noqa: I001 — set OMP/thread env before numpy

import os
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _fast_mode() -> bool:
    if "--fast" in sys.argv:
        return True
    if settings.FAST_PREPROCESS:
        return True
    return os.environ.get("DEVICESAFE_FAST_PREPROCESS", "").lower() in ("1", "true", "yes")


FAST = _fast_mode()
nlp = None
if not FAST:
    import spacy

    # en_core_sci_md is a scispaCy model — not installable via `python -m spacy download`.
    # Install: pip install scispacy
    #          pip install "en_core_sci_md @ https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz"
    print("Loading scispaCy model...")
    nlp = spacy.load("en_core_sci_md")
    print("Model loaded.")
else:
    print("Fast mode: skipping scispaCy (saves heavy CPU). tokens/pos_tags/lemmas will be empty.")

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
    if not text or nlp is None:
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

if FAST:
    print("\nSkipping scispaCy POS tagging (fast mode).")
    results = [
        {"tokens": "", "pos_tags": "", "lemmas": ""} for _ in range(len(df))
    ]
else:
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
sample = df.iloc[min(1, len(df) - 1)]
print(f"Original:  {sample['narrative_text'][:150]}...")
print(f"Cleaned:   {sample['clean_text'][:150]}...")
if not FAST:
    print(f"Tokens:    {sample['tokens'][:150]}...")
    print(f"POS tags:  {sample['pos_tags'][:150]}...")
    print(f"Lemmas:    {sample['lemmas'][:150]}...")
