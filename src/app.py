import settings  # noqa: I001 — set OMP/thread env before numpy/torch

import pickle
from pathlib import Path

import faiss
import requests
import streamlit as st

from retrieve_utils import retrieve_rag

VECTORSTORE_DIR = Path("data/vectorstore")

st.set_page_config(
    page_title="DeviceSafe NLP",
    page_icon="",
    layout="wide"
)

@st.cache_resource
def load_vectorstore():
    index = faiss.read_index(str(VECTORSTORE_DIR / "maude.index"))
    with open(VECTORSTORE_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    with open(VECTORSTORE_DIR / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embedder = settings.load_sentence_transformer()
    dim = embedder.get_sentence_embedding_dimension()
    if index.ntotal > 0 and getattr(index, "d", dim) != dim:
        raise RuntimeError(
            f"FAISS index dimension ({getattr(index, 'd', '?')}) does not match "
            f"embedding model dimension ({dim}). Rebuild data/vectorstore with "
            f"build_vectorstore.py, or set DEVICESAFE_EMBED_MODEL to the model used for the index."
        )
    return index, metadata, chunks, embedder

def ask_ollama(prompt):
    try:
        r = requests.post(
            settings.OLLAMA_URL,
            json=settings.ollama_generate_json(prompt),
            timeout=300,
        )
        return r.json().get("response", "No response from Ollama.")
    except Exception as e:
        return f"Ollama error: {e}. Make sure Ollama is running. For slow CPUs, pull a smaller model (e.g. phi3:mini) and set DEVICESAFE_OLLAMA_MODEL."

# ── sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("DeviceSafe NLP")
    st.caption("FDA Medical Device Failure Analysis")
    st.divider()
    st.markdown("**About**")
    st.caption(
        "This tool uses Retrieval-Augmented Generation (RAG) "
        "to answer questions about medical device failures "
        "using real FDA adverse event reports."
    )
    st.divider()
    st.markdown("**Pipeline**")
    st.caption("📥 FDA device failure reports")
    st.caption("🔤 scispaCy preprocessing")
    st.caption("☁️ AWS Comprehend Medical NER")
    st.caption("🧠 BERT variant analysis")
    st.caption("🗄️ FAISS vector store")
    st.caption(f"🤖 Ollama — {settings.OLLAMA_MODEL}")
    st.caption("Slow CPU? `export DEVICESAFE_OLLAMA_MODEL=phi3:mini` and `ollama pull phi3:mini`")
    st.divider()
    num_sources = st.slider("Sources to retrieve", 2, 8, 4)
    st.divider()
    st.markdown("**CS 6030 — NLP Project**")
    st.caption("Lingamuthu Kalyanasundaram")
    st.caption("Santhiya Venkatesh")

# ── main area ──────────────────────────────────────────────────────────────
st.title("DeviceSafe NLP")
st.subheader("Medical Device Failure Signal Extraction")
st.caption(
    "Ask questions about medical device failures. "
    "All answers are grounded in real FDA adverse event reports."
)

# example questions
st.markdown("**Try asking:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Knee implant failures?"):
        st.session_state.question = "What are the most common knee implant failure modes?"
with col2:
    if st.button("Pacemaker malfunctions?"):
        st.session_state.question = "What pacemaker malfunctions have been reported?"
with col3:
    if st.button("Hip implant injuries?"):
        st.session_state.question = "What injuries are associated with hip implants?"

st.divider()

# question input
question = st.text_input(
    "Ask a question about medical device failures:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What are the most common causes of cardiac defibrillator failure?"
)

if st.button("Search", type="primary") and question:
    index, metadata, chunks, embedder = load_vectorstore()

    with st.spinner("Retrieving relevant FDA reports..."):
        results, retrieval_note = retrieve_rag(
            question, index, metadata, chunks, embedder, k=num_sources
        )
    if retrieval_note:
        st.warning(retrieval_note)

    context = ""
    for i, r in enumerate(results):
        context += f"\n[Report {i+1} | Device: {r['device']} | Event: {r['event_type']}]\n{r['text']}\n"

    prompt = f"""You are DeviceSafe NLP, an AI assistant that answers questions about medical device failures using real FDA adverse event reports.

Use ONLY the following FDA report excerpts to answer the question. Do not make up information.
If the reports do not contain enough information, say so clearly.

FDA REPORT EXCERPTS:
{context}

QUESTION: {question}

ANSWER (based only on the FDA reports above):"""

    with st.spinner("Generating answer from FDA reports..."):
        answer = ask_ollama(prompt)

    # answer
    st.markdown("### Answer")
    st.success(answer)

    # sources
    st.markdown("### Sources")
    for i, r in enumerate(results):
        with st.expander(f"Report {i+1} — {r['device']} | {r['event_type']}"):
            st.caption(f"Report ID: {r['report_id']} | Relevance distance: {r['distance']}")
            st.write(r["text"])

    # add to chat history
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({"q": question, "a": answer})

# chat history
if st.session_state.get("history"):
    st.divider()
    st.markdown("### Previous questions")
    for item in reversed(st.session_state.history[:-1]):
        with st.expander(item["q"]):
            st.write(item["a"])
