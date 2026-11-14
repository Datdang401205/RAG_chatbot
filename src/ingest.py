"""Load documents, split, embed, and persist to Chroma."""

import argparse
import hashlib
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.chain import ensure_embed_model
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBED_MODEL,
    PARSE_EXTS,
    PERSIST_DIR,
    SUPPORTED_EXTENSIONS,
    get_embeddings,
)
from src.parser import cache_path, parse_to_documents


def _list_files() -> list[Path]:
    extensions = SUPPORTED_EXTENSIONS | PARSE_EXTS
    return sorted(
        p
        for p in DATA_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in extensions
        and not p.name.startswith("TEST_")
    )


def _load(path: Path) -> list[Document]:
    rel = path.relative_to(DATA_DIR).as_posix()
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []
    return [Document(page_content=text, metadata={"source": rel})]


def _chunk_id(chunk: Document, index: int) -> str:
    source = chunk.metadata.get("source", "unknown")
    page = chunk.metadata.get("page", 0)
    raw = f"{source}|{page}|{index}|{chunk.page_content}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _split(docs: list[Document], markdown: bool) -> list[Document]:
    if markdown:
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
        )
        header_docs = []
        for doc in docs:
            for chunk in header_splitter.split_text(doc.page_content):
                chunk.metadata.update(doc.metadata)
                header_docs.append(chunk)
        docs = header_docs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


def ingest(reset: bool = False) -> dict:
    """Ingest new files from papers/ into ./chroma_db and return counts."""
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        return {"files": 0, "chunks": 0, "skipped": 0, "parsed": 0, "cached": 0}

    files = _list_files()
    if not files:
        print(f"No supported documents found in {DATA_DIR}.")
        return {"files": 0, "chunks": 0, "skipped": 0, "parsed": 0, "cached": 0}

    embeddings = get_embeddings()
    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
        collection_metadata={"embed_model": EMBED_MODEL},
    )
    if reset:
        store.reset_collection()
    else:
        ensure_embed_model(store)
    existing = set(store.get().get("ids") or [])

    total = parsed = cached = skipped = failed = 0
    for path in files:
        try:
            is_parse = path.suffix.lower() in PARSE_EXTS
            if is_parse:
                if cache_path(path).exists():
                    cached += 1
                docs = parse_to_documents(path)
                parsed += bool(docs)
            else:
                docs = _load(path)

            if not docs:
                print(f"Skipped (empty or unparsed): {path.name}")
                skipped += 1
                continue

            chunks = _split(docs, is_parse)
            new_chunks, new_ids = [], []
            for index, chunk in enumerate(chunks):
                chunk_id = _chunk_id(chunk, index)
                if chunk_id in existing:
                    continue
                existing.add(chunk_id)
                new_chunks.append(chunk)
                new_ids.append(chunk_id)

            if not new_chunks:
                skipped += 1
                continue

            store.add_documents(new_chunks, ids=new_ids)
            total += len(new_chunks)
            print(f"Ingested {path.name} ({len(new_chunks)} chunks)")
        except Exception as exc:
            print(f"Failed {path.name}: {exc}")
            failed += 1

    if total == 0 and failed == 0:
        print("All documents already ingested.")
    print(
        f"Done. Added {total} chunks to {PERSIST_DIR} | "
        f"parsed: {parsed} files, cached: {cached}."
    )
    return {
        "files": len(files),
        "chunks": total,
        "skipped": skipped,
        "parsed": parsed,
        "cached": cached,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    try:
        ingest(reset=args.reset)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
