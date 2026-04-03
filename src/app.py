import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import requests

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
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return index, metadata, chunks, embedder

def retrieve(query, index, metadata, chunks, embedder, k=4):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec.astype(np.float32), k=k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "text":       chunks[idx],
            "device":     metadata[idx]["device_name"],
            "event_type": metadata[idx]["event_type"],
            "report_id":  metadata[idx]["report_id"],
            "distance":   round(float(dist), 3)
        })
    return results

def ask_ollama(prompt, model="mistral"):
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=180
        )
        return r.json().get("response", "No response from Ollama.")
    except Exception as e:
        return f"Ollama error: {e}. Make sure Ollama is running."

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
    st.caption("🤖 Ollama (Mistral) LLM")
    st.divider()
    num_sources = st.slider("Sources to retrieve", 2, 8, 4)
    st.divider()
    st.markdown("**CS 6030 — NLP Project**")
    st.caption("Lingamuthu Kalyanasundaram")
    st.caption("Santhiya Venkatesh")

# ── main area ──────────────────────────────────────────────────────────────
st.title("🔬 DeviceSafe NLP")
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
        results = retrieve(question, index, metadata, chunks, embedder, k=num_sources)

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
