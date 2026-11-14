"""LlamaParse upload, parse, and local cache."""

import hashlib
import json
from pathlib import Path

from langchain_core.documents import Document

from src.config import BASE_DIR, LLAMA_CLOUD_API_KEY, PARSE_CACHE_DIR, PARSE_TIER

CACHE_DIR = BASE_DIR / PARSE_CACHE_DIR


def cache_path(path: Path) -> Path:
    """Return the cache file for a document's bytes."""
    digest = hashlib.sha1(path.read_bytes()).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def _from_cache(path: Path, cache: Path) -> list[Document]:
    data = json.loads(cache.read_text(encoding="utf-8"))
    return [
        Document(
            page_content=page["markdown"],
            metadata={"source": data.get("source", path.name), "page": page["page"]},
        )
        for page in data.get("pages", [])
        if page.get("markdown")
    ]


def parse_to_documents(path: Path) -> list[Document]:
    """Parse a file with LlamaParse or load its cached pages."""
    cache = cache_path(path)
    if cache.exists():
        return _from_cache(path, cache)

    if not LLAMA_CLOUD_API_KEY:
        print(f"Warning: LLAMA_CLOUD_API_KEY is missing; skipping {path.name}.")
        return []

    try:
        from llama_cloud import LlamaCloud

        client = LlamaCloud()
        file = client.files.create(file=str(path), purpose="parse")
        result = client.parsing.parse(
            file_id=file.id,
            tier=PARSE_TIER,
            version="latest",
            expand=["markdown"],
        )
        markdown = getattr(result, "markdown", None)
        pages = [
            {"page": page.page_number, "markdown": page.markdown}
            for page in getattr(markdown, "pages", [])
            if page.success and page.markdown
        ]
    except Exception as exc:
        print(f"Warning: LlamaParse failed for {path.name}: {exc}")
        return []

    if not pages:
        print(f"Warning: LlamaParse returned no pages for {path.name}.")
        return []

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps({"source": path.name, "pages": pages}, ensure_ascii=False),
        encoding="utf-8",
    )
    return [
        Document(
            page_content=page["markdown"],
            metadata={"source": path.name, "page": page["page"]},
        )
        for page in pages
    ]
