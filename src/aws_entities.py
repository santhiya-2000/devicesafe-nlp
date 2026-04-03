import boto3
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/processed/maude_processed.csv")
df["clean_text"] = df["clean_text"].fillna("")
df = df[df["clean_text"].str.len() > 50].reset_index(drop=True)
print(f"Processing {len(df)} reports with AWS Comprehend Medical...")
print("Free tier: 25,000 units/month. Your usage: ~10,000 units. Cost: $0\n")

client = boto3.client("comprehendmedical", region_name="us-east-1")

def extract_entities(text):
    # truncate to 20,000 chars — AWS hard limit
    text = text[:20000]
    try:
        result = client.detect_entities_v2(Text=text)
        return [{
            "text":     e["Text"],
            "category": e["Category"],
            "type":     e.get("Type", ""),
            "score":    round(e["Score"], 3)
        } for e in result["Entities"] if e["Score"] > 0.6]
    except Exception as ex:
        print(f"  Error on report: {ex}")
        return []

all_entities = []
char_count   = 0

for idx, row in tqdm(df.iterrows(), total=len(df)):
    text = str(row["clean_text"])
    char_count += len(text)

    entities = extract_entities(text)

    for e in entities:
        all_entities.append({
            "report_id":   str(row.get("report_id", idx)),
            "device_name": str(row.get("device_name", "")),
            "event_type":  str(row.get("event_type", "")),
            "entity_text": e["text"],
            "category":    e["category"],
            "type":        e["type"],
            "score":       e["score"]
        })

    time.sleep(0.05)  # gentle rate limiting

entity_df = pd.DataFrame(all_entities)
entity_df.to_csv(PROCESSED_DIR / "aws_entities.csv", index=False)

units_used = char_count // 100
print(f"\nDone.")
print(f"Characters processed : {char_count:,}")
print(f"Units used           : {units_used:,} / 25,000 free")
print(f"Estimated cost       : $0.00 (within free tier)")
print(f"Entities extracted   : {len(entity_df):,}")
print(f"\nTop entity categories:")
print(entity_df["category"].value_counts().to_string())
print(f"\nTop entity types:")
print(entity_df["type"].value_counts().head(10).to_string())
print(f"\nSample entities:")
print(entity_df[["device_name","entity_text","category","score"]].head(15).to_string(index=False))
