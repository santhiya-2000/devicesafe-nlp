"""
Runtime tuning for low-CPU machines. Import this module before sentence_transformers/torch.

Environment (all optional):
  DEVICESAFE_EMBED_MODEL     — SentenceTransformer name (must match vectorstore build; default all-MiniLM-L6-v2)
  DEVICESAFE_OLLAMA_MODEL    — Ollama model tag (default mistral; try phi3:mini or llama3.2:1b if CPU-bound)
  DEVICESAFE_OLLAMA_URL      — generate API URL
  DEVICESAFE_OLLAMA_MAX_TOKENS — cap LLM output length for faster replies (default 384)
  DEVICESAFE_TORCH_THREADS   — PyTorch intra-op threads (default 2; use 1 on dual-core)
  DEVICESAFE_ENCODE_BATCH    — batch size when building embeddings (default 8)
  DEVICESAFE_FAST_PREPROCESS — 1 = skip scispaCy in preprocess.py (much faster; RAG only needs clean_text)
  DEVICESAFE_MAX_PER_TERM    — FDA records per search term in download_data.py (default 200)
"""
from __future__ import annotations

import os


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


# Limit BLAS/OpenMP thread explosion before NumPy/PyTorch load (common freeze cause on laptops).
def _apply_thread_env() -> None:
    for key, val in (
        ("OMP_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
    ):
        os.environ.setdefault(key, val)


_apply_thread_env()

EMBED_MODEL = os.environ.get("DEVICESAFE_EMBED_MODEL", "all-MiniLM-L6-v2")
OLLAMA_MODEL = os.environ.get("DEVICESAFE_OLLAMA_MODEL", "mistral")
OLLAMA_URL = os.environ.get("DEVICESAFE_OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MAX_TOKENS = max(32, _int("DEVICESAFE_OLLAMA_MAX_TOKENS", 384))
TORCH_THREADS = max(1, _int("DEVICESAFE_TORCH_THREADS", 2))
ENCODE_BATCH = max(1, _int("DEVICESAFE_ENCODE_BATCH", 8))
FAST_PREPROCESS = os.environ.get("DEVICESAFE_FAST_PREPROCESS", "").lower() in ("1", "true", "yes")
MAX_PER_TERM = max(1, _int("DEVICESAFE_MAX_PER_TERM", 200))


def configure_torch_threads() -> None:
    """Call after torch is imported."""
    try:
        import torch

        torch.set_num_threads(TORCH_THREADS)
        torch.set_num_interop_threads(1)
    except ImportError:
        pass


def load_sentence_transformer():
    """Load embedder with CPU-friendly thread settings."""
    from sentence_transformers import SentenceTransformer

    configure_torch_threads()
    return SentenceTransformer(EMBED_MODEL)


def ollama_generate_json(prompt: str) -> dict:
    """Payload for Ollama /api/generate."""
    return {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": OLLAMA_MAX_TOKENS,
        },
    }
