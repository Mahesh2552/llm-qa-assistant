from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import (
    ExactMatchFilter,
    MetadataFilters,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import settings


@dataclass(frozen=True)
class RetrievedSnippet:
    text: str
    score: float | None
    metadata: dict


class VectorRetriever:
    def __init__(
        self,
        *,
        persist_dir: Path | None = None,
        collection_name: str | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.persist_dir = persist_dir or settings.VECTOR_STORE_DIR
        self.collection_name = collection_name or settings.CHROMA_COLLECTION

        client = chromadb.PersistentClient(path=str(self.persist_dir))
        collection = client.get_or_create_collection(self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=embed_model,
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        project_name: str | None = None,
    ) -> list[RetrievedSnippet]:
        top_k = top_k or settings.VECTOR_TOP_K

        filters = None
        if project_name:
            # Use LlamaIndex metadata filter objects (required by ChromaVectorStore integration)
            filters = MetadataFilters(filters=[ExactMatchFilter(key="project_name", value=project_name)])

        retriever = self.index.as_retriever(similarity_top_k=top_k, filters=filters)
        nodes: list[NodeWithScore] = retriever.retrieve(query)

        snippets: list[RetrievedSnippet] = []
        for n in nodes:
            node = n.node
            snippets.append(
                RetrievedSnippet(
                    text=node.get_content(metadata_mode="none"),
                    score=float(n.score) if n.score is not None else None,
                    metadata=dict(node.metadata or {}),
                )
            )
        return snippets

    @staticmethod
    def snippets_to_context(snippets: list[RetrievedSnippet], *, max_chars: int = 6000) -> str:
        parts: list[str] = []
        total = 0
        for s in snippets:
            header = []
            if s.metadata.get("project_name"):
                header.append(f"project={s.metadata.get('project_name')}")
            if s.metadata.get("section"):
                header.append(f"section={s.metadata.get('section')}")
            header_str = f"[{', '.join(header)}]" if header else "[snippet]"

            block = f"{header_str}\n{s.text.strip()}\n"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n---\n".join(parts).strip()

