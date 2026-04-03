import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

df = pd.read_csv(PROCESSED_DIR / "aws_entities.csv")
print(f"Total entities : {len(df)}")
print(f"Reports covered: {df['report_id'].nunique()}")

def get_top(group, category, n=3):
    matches = group[group["category"] == category]["entity_text"]
    return " | ".join(matches.unique()[:n]) if len(matches) > 0 else ""

report_summary = []
for report_id, group in df.groupby("report_id"):
    row = group.iloc[0]
    report_summary.append({
        "report_id":   report_id,
        "device_name": row["device_name"],
        "event_type":  row["event_type"],
        "conditions":  get_top(group, "MEDICAL_CONDITION"),
        "anatomy":     get_top(group, "ANATOMY"),
        "procedures":  get_top(group, "TEST_TREATMENT_PROCEDURE"),
        "medications": get_top(group, "MEDICATION"),
    })

summary_df = pd.DataFrame(report_summary)
summary_df.to_csv(PROCESSED_DIR / "entity_summary.csv", index=False)

print(f"\nStructured entity table — {len(summary_df)} reports")
print(f"\nSample:")
print(summary_df[["device_name","conditions","anatomy","procedures"]].head(10).to_string(index=False))

print(f"\nTop medical conditions across all reports:")
all_conditions = df[df["category"]=="MEDICAL_CONDITION"]["entity_text"]
print(all_conditions.value_counts().head(15).to_string())

print(f"\nTop anatomical sites:")
all_anatomy = df[df["category"]=="ANATOMY"]["entity_text"]
print(all_anatomy.value_counts().head(10).to_string())

print(f"\nTop procedures:")
all_procs = df[df["category"]=="TEST_TREATMENT_PROCEDURE"]["entity_text"]
print(all_procs.value_counts().head(10).to_string())
