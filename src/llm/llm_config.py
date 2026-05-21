from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from dotenv import load_dotenv
from llama_index.core.base.llms.types import LLMMetadata, MessageRole
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI as OpenAILlm

GROQ_OPENAI_COMPAT_BASE = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
# Groq Llama 3.x production models use 128k context; override via GROQ_CONTEXT_WINDOW if needed.
DEFAULT_GROQ_CONTEXT_WINDOW = 131_072


class _GroqOpenAILLM(OpenAILlm):
    """
    Groq uses an OpenAI-compatible API, but LlamaIndex's OpenAI LLM only allows OpenAI model IDs
    when resolving context window and tiktoken. Subclassing skips that so Groq ids work.
    """

    @property
    def _tokenizer(self) -> Optional[Any]:
        return None

    @property
    def metadata(self) -> LLMMetadata:
        raw = os.getenv("GROQ_CONTEXT_WINDOW", "").strip()
        context_window = int(raw) if raw else DEFAULT_GROQ_CONTEXT_WINDOW
        return LLMMetadata(
            context_window=context_window,
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name=self.model,
            system_role=MessageRole.SYSTEM,
        )


@dataclass(frozen=True)
class LLMInfo:
    provider: str
    model: str


def get_llm() -> tuple[LLM, LLMInfo]:
    """
    Auto-select an LLM.

    Priority:
    1) Groq if GROQ_API_KEY is set (OpenAI-compatible API; no local RAM for weights)
    2) OpenAI if OPENAI_API_KEY is set
    3) Ollama local otherwise
    """
    load_dotenv()

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL
        api_base = os.getenv("GROQ_API_BASE", GROQ_OPENAI_COMPAT_BASE).strip() or GROQ_OPENAI_COMPAT_BASE
        llm: LLM = _GroqOpenAILLM(
            model=model,
            api_key=groq_key,
            api_base=api_base,
            temperature=0.2,
            timeout=120.0,
        )
        return llm, LLMInfo(provider="groq", model=model)

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
        llm = OpenAILlm(model=model, api_key=openai_key, temperature=0.2)
        return llm, LLMInfo(provider="openai", model=model)

    from llama_index.llms.ollama import Ollama

    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip() or "llama3.2:3b"
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip() or "http://localhost:11434"
    llm = Ollama(model=model, base_url=base_url, temperature=0.2, request_timeout=120.0)
    return llm, LLMInfo(provider="ollama", model=model)

