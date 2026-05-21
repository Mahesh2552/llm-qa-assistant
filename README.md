# Oak Training Assistant

Internal “ChatGPT for company projects” using **Hybrid Retrieval (GraphRAG + Vector RAG)**:

- **GraphRAG**: ensures **complete project context** (especially full workflows/steps) is always retrieved.
- **Vector RAG (ChromaDB)**: brings in supporting snippets for nuanced questions and comparisons.

## What you get

- Data ingestion pipeline: `data/` → **graph_store** + **vector_store**
- Hybrid retrieval: project detection → graph retrieval → vector retrieval → LLM reasoning → structured answer
- Streamlit chat UI

## Prerequisites

- **Python 3.12+** recommended (latest stable).
- **OpenAI API key** (recommended): set `OPENAI_API_KEY`
  - Alternative: **Ollama** (local) is supported as a fallback.

## Install

From the repository root:

```bash
cd oak_training_assistant
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Optional env file:

- Copy `oak_training_assistant/.env.example` → `oak_training_assistant/.env`
- Set:
  - `OPENAI_API_KEY=...`
  - Optional: `OPENAI_MODEL=gpt-4.1-mini`

## Add project documents

Drop YAML files in `data/projects/`. See the included example:

- `data/projects/raman_drug_detection.yaml`

## Build indexes (vector + graph)

```bash
python scripts/run_ingestion.py
```

This writes:

- `vector_store/` (ChromaDB persistent store)
- `graph_store/oak_graph.json` (project knowledge graph)

## Run the chat app

```bash
streamlit run src/app/streamlit_app.py
```

## Notes on the Hybrid Retrieval design

- **Project detection** identifies the most likely project name (fuzzy match + metadata hints).
- **Graph retrieval** returns **all fields** and **all workflow steps** for that project (no chunk-loss).
- **Vector retrieval** adds supporting excerpts across docs, filtered by project when possible.

## Extending

1. Add new YAML docs to `data/projects/`
2. Re-run ingestion: `python scripts/run_ingestion.py`
3. Restart Streamlit (if needed)
