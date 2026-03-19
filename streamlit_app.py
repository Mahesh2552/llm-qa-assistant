"""Streamlit web UI for the flan-t5-based Document QA assistant.

This uses the shared backend in doc_qa_backend.py, but with a browser interface.
"""

import streamlit as st
from sentence_transformers import SentenceTransformer

from doc_qa_backend import load_index, retrieve, build_prompt, EMB_DIR, TOP_K
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


st.set_page_config(page_title="Document QA Assistant", layout="centered")
st.title("Document QA Assistant")
# st.caption("Ask questions over your indexed documents using retrieval + flan-t5-base.")


@st.cache_resource
def load_models():
    """Load embeddings, embedding model and flan-t5-base once."""
    vectors, metadata = load_index()
    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return vectors, metadata, emb_model, tokenizer, model


with st.spinner(f"Loading index from {EMB_DIR} and models… (first run may take 1–2 minutes)"):
    try:
        vectors, metadata, emb_model, tokenizer, model = load_models()
    except Exception as e:
        st.error(f"Failed to load models or index: {e}")
        st.stop()

st.sidebar.header("Settings")
top_k = st.sidebar.slider("Chunks to use as context", 2, 10, TOP_K)

question = st.text_input("Your question", placeholder="e.g. How many casual leaves are available?")
if not question:
    st.info("Enter a question above to get an answer.")
    st.stop()

with st.spinner("Searching documents..."):
    results = retrieve(question, emb_model, vectors, metadata, top_k=top_k)
    context_chunks = [r[1]["text"] for r in results]

with st.spinner("Generating answer with flan-t5-base..."):
    prompt = build_prompt(context_chunks, question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_length=256,
        num_beams=4,
        max_new_tokens=256,
        do_sample=False,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

st.subheader("Answer")
st.write(answer)

st.subheader("Top context snippet")
if context_chunks:
    st.text(context_chunks[0].strip())

st.subheader("Sources")
for score, meta in results:
    st.caption(f"**{meta['source']}** (score: {score:.3f})")

