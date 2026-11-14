"""Retriever, prompt, and streaming RAG chain."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

from src.config import (
    COLLECTION_NAME,
    EMBED_MODEL,
    PERSIST_DIR,
    RETRIEVER_K,
    get_embeddings,
    get_llm,
)

SYSTEM_PROMPT = """You are a question-answering assistant.
Answer the question using ONLY the context below. Do not use outside knowledge.
If the answer is not in the context, reply exactly: I don't know.
When you answer, cite the source filenames you used, like [filename].

Context:
{context}"""


def _format_docs(docs: list[Document]) -> str:
    if not docs:
        return "(no relevant context found)"
    parts = []
    for doc in docs:
        name = Path(doc.metadata.get("source", "unknown")).name
        parts.append(f"[{name}]\n{doc.page_content}")
    return "\n\n".join(parts)


def ensure_embed_model(store: Chroma) -> None:
    """Fail if the persisted vectors were built with another embedding model."""
    stored = (store._collection.metadata or {}).get("embed_model")
    if stored != EMBED_MODEL:
        raise RuntimeError(
            "Embedding model changed: run `python -m src.ingest --reset`"
        )


def get_vectorstore(check_embedding: bool = True) -> Chroma:
    """Create the persisted Chroma store."""
    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
        embedding_function=get_embeddings(),
        collection_metadata={"embed_model": EMBED_MODEL},
    )
    if check_embedding:
        ensure_embed_model(store)
    return store


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents with their source filenames."""
    return _format_docs(docs)


def build_answer_chain():
    """Build prompt + streaming LLM chain expecting context, history, question."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )
    llm = get_llm()
    return prompt | llm


def build_rag_chain():
    """Build retriever + prompt + LLM chain with streaming output."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    return (
        RunnablePassthrough.assign(
            context=(lambda x: x["question"]) | retriever | _format_docs
        )
        | prompt
        | llm
        | StrOutputParser()
    )
