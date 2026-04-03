import requests
import pandas as pd
from pathlib import Path
import time

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api.fda.gov/device/event.json"

SEARCH_TERMS = [
    "knee+implant",
    "pacemaker",
    "hip+implant",
    "spinal+implant",
    "cardiac+defibrillator"
]

def fetch_reports(search_term, total=200):
    records = []
    limit   = 100
    skip    = 0
    while len(records) < total:
        params = {
            "search": f"device.generic_name:{search_term}",
            "limit":  limit,
            "skip":   skip
        }
        try:
            r = requests.get(BASE_URL, params=params, timeout=15)
            if r.status_code != 200:
                print(f"  Stopped at {len(records)} records (status {r.status_code})")
                break
            results = r.json().get("results", [])
            if not results:
                break
            records.extend(results)
            skip += limit
            time.sleep(0.5)
        except Exception as e:
            print(f"  Error: {e}")
            break
    return records[:total]

def extract_fields(record):
    # try all text type codes, not just D and N
    narratives = record.get("mdr_text", [])
    text = " ".join(
        t.get("text", "") for t in narratives
        if isinstance(t, dict) and t.get("text")
    )

    devices     = record.get("device", [{}])
    device_name = devices[0].get("generic_name", "") if devices else ""
    brand_name  = devices[0].get("brand_name", "")   if devices else ""
    model_num   = devices[0].get("model_number", "")  if devices else ""

    patient_outcomes = []
    for p in record.get("patient", []):
        outcome = p.get("sequence_number_outcome", "")
        if isinstance(outcome, list):
            patient_outcomes.extend([str(o) for o in outcome])
        elif outcome:
            patient_outcomes.append(str(outcome))

    return {
        "report_id":        str(record.get("report_number", "")),
        "date":             str(record.get("date_received", "")),
        "device_name":      str(device_name),
        "brand_name":       str(brand_name),
        "model_number":     str(model_num),
        "event_type":       str(record.get("event_type", "")),
        "narrative_text":   text.strip(),
        "patient_outcomes": ", ".join(patient_outcomes)
    }

all_records = []

for term in SEARCH_TERMS:
    print(f"Fetching: {term}...")
    raw = fetch_reports(term, total=200)
    print(f"  Got {len(raw)} raw reports")

    # debug: print what text fields look like in first record
    if raw:
        sample = raw[0]
        print(f"  Sample mdr_text: {sample.get('mdr_text', 'NONE')[:2]}")

    extracted = [extract_fields(r) for r in raw]
    all_records.extend(extracted)
    time.sleep(1)

df = pd.DataFrame(all_records)

# fill any NaN with empty string before filtering
df["narrative_text"] = df["narrative_text"].fillna("")

print(f"\nBefore filter: {len(df)} reports")
print(f"Reports with text > 50 chars: {(df['narrative_text'].str.len() > 50).sum()}")
print(f"Avg text length: {int(df['narrative_text'].str.len().mean())} chars")

# keep everything with any text at all first so we can see what we have
df_all = df[df["narrative_text"].str.len() > 0]
df_filtered = df[df["narrative_text"].str.len() > 50]

print(f"\nSample narrative from first record with any text:")
if len(df_all) > 0:
    print(df_all["narrative_text"].iloc[0][:300])
else:
    print("NO NARRATIVES FOUND - checking raw structure...")
    if all_records:
        print("First raw record keys:", list(raw[0].keys()) if raw else "empty")

df_filtered.to_csv(RAW_DIR / "maude_reports.csv", index=False)
print(f"\nSaved {len(df_filtered)} reports to data/raw/maude_reports.csv")
