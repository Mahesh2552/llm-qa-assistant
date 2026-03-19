"""Backend for Document QA app: retrieval + flan-t5-base prompt building.

Provides: load_index(), retrieve(), build_prompt().
Used by streamlit_app.py (and can be reused by a CLI).
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Use local embeddings folder
EMB_DIR = (Path(__file__).resolve().parent / "embeddings").resolve()
TOP_K = 5


def load_index():
    """Load vectors and metadata from ./embeddings/."""
    vectors_path = EMB_DIR / "vectors.npy"
    meta_path = EMB_DIR / "metadata.json"

    if not vectors_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found in {EMB_DIR}. Run 'python build_index.py' in the project root first."
        )

    vectors = np.load(vectors_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return vectors, metadata


def retrieve(
    query: str,
    emb_model: SentenceTransformer,
    vectors: np.ndarray,
    metadata: List[dict],
    top_k: int = TOP_K,
) -> List[Tuple[float, dict]]:
    """Return top-k (score, chunk_meta) by cosine similarity."""
    query_emb = emb_model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(query_emb, vectors)[0]
    top_idx = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), metadata[int(i)]) for i in top_idx]


def build_prompt(context_chunks: List[str], question: str) -> str:
    """Build a prompt for flan-t5-base: context + question."""
    context = "\n\n".join(context_chunks)
    prompt = (
        "You are a helpful assistant. "
        "Answer the question based ONLY on the context below. "
        "If the answer is not in the context, reply exactly with: not in the context\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return prompt

