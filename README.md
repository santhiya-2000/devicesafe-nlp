# DeviceSafe NLP

A comprehensive NLP pipeline for analyzing medical device adverse event reports from the FDA's MAUDE (Manufacturer and User Facility Device Experience) database. This project enables intelligent querying and analysis of device safety data using modern NLP techniques including named entity recognition, vector embeddings, and retrieval-augmented generation (RAG).

## Features

- **Data Acquisition**: Automated download of MAUDE reports from FDA API
- **Preprocessing**: Text cleaning and normalization with optional biomedical NLP processing
- **Entity Extraction**: Multiple NER models for identifying medical entities (diseases, devices, etc.)
- **Vector Store**: FAISS-based semantic search with sentence transformers
- **Chatbot Interface**: Streamlit web app for interactive querying
- **Evaluation**: Comparative analysis of different NER approaches

## Project Structure

```
├── data/
│   ├── raw/           # Raw MAUDE data downloads
│   ├── processed/     # Cleaned and processed data
│   └── vectorstore/   # FAISS indices and metadata
├── notebooks/         # Jupyter notebooks for analysis
├── src/               # Python source code
│   ├── app.py         # Streamlit chatbot application
│   ├── preprocess.py  # Data cleaning pipeline
│   ├── build_vectorstore.py  # Embedding and indexing
│   ├── bert_*.py      # NER model implementations
│   └── *.py           # Utility and evaluation scripts
└── README.md
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd devicesafe-nlp
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install torch sentence-transformers transformers pandas numpy faiss-cpu requests streamlit tqdm scispacy
   ```

   For biomedical NLP processing:
   ```bash
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz
   python -m spacy download en_core_web_sm
   ```

4. **Set up Ollama** (for local LLM):
   ```bash
   # Install Ollama from https://ollama.ai/
   ollama pull mistral  # or another model like llama3.2:1b
   ```

## Usage

1. **Download MAUDE data**:
   ```bash
   python src/download_data.py
   ```

2. **Preprocess the data**:
   ```bash
   python src/preprocess.py
   ```

3. **Build vector store**:
   ```bash
   python src/build_vectorstore.py
   ```

4. **Run the chatbot**:
   ```bash
   streamlit run src/app.py
   ```

## Configuration

The application can be configured via environment variables:

- `DEVICESAFE_EMBED_MODEL`: Sentence transformer model (default: all-MiniLM-L6-v2)
- `DEVICESAFE_OLLAMA_MODEL`: Ollama model (default: mistral)
- `DEVICESAFE_OLLAMA_URL`: Ollama API URL
- `DEVICESAFE_MAX_TOKENS`: Maximum LLM response length
- `DEVICESAFE_TORCH_THREADS`: PyTorch thread count for CPU optimization

## Data Sources

- **MAUDE Database**: FDA's Manufacturer and User Facility Device Experience database
- **API Endpoint**: https://api.fda.gov/device/event.json

## Dependencies

- Python 3.8+
- PyTorch
- Sentence Transformers
- Transformers (Hugging Face)
- FAISS
- Streamlit
- Pandas, NumPy
- Requests
- TQDM
- SciSpaCy (optional, for biomedical preprocessing)

