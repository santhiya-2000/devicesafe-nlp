# src/check.py
import spacy
import transformers
import faiss
import streamlit
import langchain

print("spaCy:", spacy.__version__)
print("transformers:", transformers.__version__)
print("faiss imported OK")
print("streamlit:", streamlit.__version__)
print("langchain:", langchain.__version__)

nlp = spacy.load("en_core_sci_md")
doc = nlp("The knee implant fractured during surgery.")
for token in doc:
    print(f"{token.text:20} {token.pos_:10} {token.lemma_}")