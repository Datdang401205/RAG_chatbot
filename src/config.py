"""Environment and shared constants."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH, override=True)

DATA_DIR = BASE_DIR / "papers"
PERSIST_DIR = BASE_DIR / "chroma_db"

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "").strip().strip("\"'").strip()

COHERE_MODEL = os.getenv("COHERE_MODEL", "command-a-03-2025").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5").strip()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
RETRIEVER_K = 4
COLLECTION_NAME = "rag"
HISTORY_TURNS = 4
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
PARSE_EXTS = {".pdf", ".docx", ".pptx", ".xlsx"}
PARSE_CACHE_DIR = ".parse_cache"
PARSE_TIER = "cost_effective"


def require_api_key() -> str:
    """Return the cleaned Cohere key or fail fast."""
    key = os.getenv("COHERE_API_KEY", "").strip().strip("\"'").strip()
    if not key:
        raise RuntimeError("COHERE_API_KEY missing")
    return key


def get_llm() -> ChatCohere:
    """Create the Cohere chat model."""
    return ChatCohere(
        cohere_api_key=require_api_key(),
        model=COHERE_MODEL,
        temperature=0,
    )


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create the local HuggingFace embedding model (no API key needed)."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
