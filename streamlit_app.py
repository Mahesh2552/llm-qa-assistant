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


GEN_MODEL_NAME = "google/flan-t5-small"


@st.cache_resource
def load_models():
    """Load embeddings, embedding model and generator once."""
    vectors, metadata = load_index()
    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
    return vectors, metadata, emb_model, tokenizer, model


with st.spinner(f"Loading index from {EMB_DIR} and models… (first run may take 1–2 minutes)"):
    try:
        vectors, metadata, emb_model, tokenizer, model = load_models()
    except Exception as e:
        st.error(f"Failed to load models or index: {e}")
        st.stop()

st.sidebar.header("Try asking questions like")
st.sidebar.markdown(
    """
Questions this assistant can answer from your documents:

- **Leaves:** How many casual leaves are available per year? How many sick leaves do we get?
- **Working hours:** What are the standard working hours (and flexible hours rules)?
- **Remote work:** Can employees work from home? If yes, for how many days per week?
- **Holidays (2025):** What are the public holiday dates?
- **Company policies:** What is the advance notice required for planned leave?
- **Product FAQ:** What does the analytics product do? How frequently is data refreshed?
- **Data analytics tips:** What data quality checks should I run before analysis?
"""
)

# Use the backend default for context size (keeps the UI simple and faster on Free tiers).
top_k = TOP_K

question = st.text_input(
    "Your question",
    placeholder="e.g. How many casual leaves are available per year?",
)
if not question:
    st.info("Enter a question above to get an answer.")
    st.stop()

with st.spinner("Searching documents..."):
    results = retrieve(question, emb_model, vectors, metadata, top_k=top_k)
    context_chunks = [r[1]["text"] for r in results]

with st.spinner("Generating answer..."):
    prompt = build_prompt(context_chunks, question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_new_tokens=96,
        num_beams=1,
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
