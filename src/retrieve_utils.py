"""Dense retrieval with query expansion and domain-keyword reranking for device-specific questions."""

from __future__ import annotations

from typing import Optional

import numpy as np

# When the question names a device family, prefer chunks whose text or device field mentions
# related terms (reduces dental / generic "implant" false positives).
_DOMAIN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "knee": (
        "knee",
        "tibial",
        "femoral",
        "patella",
        "patello",
        "unicompartmental",
        "arthroplasty",
        "tka",
        "ukr",
        "meniscal",
    ),
    "hip": ("hip", "acetabular", "femoral head", "femoral stem", "arthroplasty", "th"),
    "pacemaker": ("pacemaker", "pacing", "atrial", "ventricular", "crt", "lead"),
    "defibrillator": ("defibrillator", "icd", "cardioverter", "crt-d", "tachy"),
    "spinal": ("spinal", "spine", "vertebr", "disc", "pedicle", "lumbar", "cervical", "sacral"),
    "cardiac": ("cardiac", "heart", "coronary", "valve", "stent", "aortic"),
}


def _active_domains(question: str) -> list[str]:
    q = question.lower()
    return [d for d in _DOMAIN_SYNONYMS if d in q]


def _embedding_query_text(question: str) -> str:
    """Bias dense search toward device-specific language instead of generic 'implant'."""
    parts = [question]
    for d in _active_domains(question):
        parts.append(" ".join(_DOMAIN_SYNONYMS[d]))
    return " ".join(parts) if len(parts) > 1 else question


def _matches_domain(text: str, device: str, domains: list[str]) -> bool:
    if not domains:
        return True
    blob = f"{device} {text}".lower()
    for d in domains:
        if any(s in blob for s in _DOMAIN_SYNONYMS[d]):
            return True
    return False


def retrieve_rag(
    query: str,
    index,
    metadata: list,
    chunks: list,
    embedder,
    k: int = 4,
    oversample: int = 6,
) -> tuple[list[dict], Optional[str]]:
    """
    Return (results, optional_warning). Uses expanded embedding + keyword rerank when the
    question names a device domain (knee, hip, ...).
    """
    embed_q = _embedding_query_text(query)
    query_vec = embedder.encode([embed_q], convert_to_numpy=True)
    domains = _active_domains(query)
    n = int(index.ntotal)
    k_fetch = min(max(k * oversample, 32 if domains else k * oversample), n)
    distances, indices = index.search(query_vec.astype(np.float32), k=k_fetch)

    rows: list[dict] = []
    for dist, idx in zip(distances[0], indices[0]):
        text = chunks[idx]
        device = str(metadata[idx].get("device_name", ""))
        rows.append(
            {
                "text": text,
                "device": device,
                "event_type": str(metadata[idx].get("event_type", "")),
                "report_id": str(metadata[idx].get("report_id", "")),
                "distance": round(float(dist), 3),
                "_match": _matches_domain(text, device, domains),
            }
        )

    note: Optional[str] = None
    if domains:
        preferred = [r for r in rows if r["_match"]]
        if preferred:
            chosen = preferred[:k]
        else:
            chosen = rows[:k]
            note = (
                "No chunk in the broadened search clearly matched your device terms; "
                "showing the closest semantic matches—they may be off-topic."
            )
    else:
        chosen = rows[:k]

    for r in chosen:
        r.pop("_match", None)
    return chosen, note


def retrieval_domain_stats(question: str, results: list[dict]) -> dict:
    """
    Automatic retrieval quality signal: share of retrieved chunks whose text/device matches
    domain keywords implied by the question (see _DOMAIN_SYNONYMS).
    """
    domains = _active_domains(question)
    if not domains or not results:
        return {
            "domain_query": bool(domains),
            "domain_matches": 0,
            "domain_match_rate": float("nan"),
        }
    n_ok = sum(1 for r in results if _matches_domain(r["text"], r["device"], domains))
    return {
        "domain_query": True,
        "domain_matches": n_ok,
        "domain_match_rate": n_ok / len(results),
    }
