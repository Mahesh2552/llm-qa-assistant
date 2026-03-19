## Document QA Assistant

A small Streamlit app that answers questions using retrieval over your local documents (PDF/TXT).

## What it does

1. Loads and chunks documents from `./data/` (`.txt` and `.pdf`).
2. Creates embeddings for each chunk and stores them in `./embeddings/` (`vectors.npy`, `metadata.json`).
3. For each question:
   - Retrieves the top `k` most similar chunks using cosine similarity (via `all-MiniLM-L6-v2` embeddings).
   - Builds a prompt from the retrieved context.
   - Generates an answer with `flan-t5-base`.

If the answer is not found in the retrieved context, the model is instructed to reply exactly:
`not in the context`

## Project layout

- `streamlit_app.py`: Streamlit UI (question input, answer display, and sources)
- `doc_qa_backend.py`: Retrieval logic (`load_index`, `retrieve`) and prompt builder (`build_prompt`)
- `build_index.py`: One-time script to build `embeddings/` from `data/`
- `data/`: Your input documents (`*.txt`, `*.pdf`)
- `embeddings/`: Precomputed embeddings artifacts (`vectors.npy`, `metadata.json`)

## Setup

### 1. Create/activate a virtual environment

Windows PowerShell (example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

## Build (or rebuild) the embeddings index

Run this when you add/change documents in `./data/`.

```powershell
python build_index.py
```

Outputs:

- `embeddings/vectors.npy`
- `embeddings/metadata.json`

## Run the app

```powershell
streamlit run streamlit_app.py
```

On the first run, loading the models (and/or index) may take a minute or two.

## Configuration

In the Streamlit sidebar:

- `Chunks to use as context` (top-k): controls how many retrieved chunks are included in the prompt.

## Notes / limitations

- Retrieval is chunk-based and uses vector similarity; it may miss information if chunks are too small/large or the question doesn’t match wording well.
- The app only answers based on retrieved context (per the prompt). There is no web search.
- `build_index.py` uses character-based chunking with:
  - `CHUNK_SIZE = 500`
  - `CHUNK_OVERLAP = 100`

