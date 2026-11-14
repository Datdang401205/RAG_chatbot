"""Streamlit UI for the RAG chatbot."""

import time
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src import chain, ingest
from src.config import COHERE_MODEL, HISTORY_TURNS, require_api_key

st.set_page_config(page_title="RAG Chatbot", layout="wide")

QUESTIONS = [
    "Who is the CISO?",
    "How many days of annual leave do employees get?",
    "Which plan has the Pilot AI assistant, and what is its price with annual billing?",
    "What is the API rate limit for Enterprise?",
    "How fast must we respond to a SEV-1 incident?",
    "Who is the CTO of Zentora Labs?",
    "Is Zentora Labs headquartered in Lisbon?",
    "What is the free trial length for Enterprise?",
]

try:
    require_api_key()
except Exception as exc:
    st.error(f"{exc} Fix: copy .env.example to .env and set COHERE_API_KEY.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None


@st.cache_resource(show_spinner=False)
def get_resources(k: int):
    store = chain.get_vectorstore()
    has_docs = bool(store.get(limit=1).get("ids"))
    return store, chain.build_answer_chain(), has_docs


def _status_code(exc: Exception) -> int | None:
    return getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )


def _is_429(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        _status_code(exc) == 429
        or "429" in text
        or "rate limit" in text
        or "too many requests" in text
    )


def render_meta(sources, latency, usage):
    if sources:
        with st.expander(f"Sources ({len(sources)} chunks)"):
            for src in sources:
                st.markdown(f"**{src['name']}** — score: {src['score']:.4f}")
                st.caption(src["text"][:300])
    if usage.get("input_tokens") is None and usage.get("output_tokens") is None:
        st.caption(f"Latency: {latency:.2f}s")
    else:
        st.caption(
            f"Latency: {latency:.2f}s | Tokens: "
            f"{usage.get('input_tokens', 0)} prompt / "
            f"{usage.get('output_tokens', 0)} completion"
        )


with st.sidebar:
    st.header("Controls")
    st.caption(f"Model: {COHERE_MODEL}")

    if st.button("Ingest papers/", use_container_width=True):
        with st.status("Ingesting...", expanded=True) as status:
            try:
                summary = ingest.ingest()
                st.write(
                    f"Files: {summary['files']} | chunks added: "
                    f"{summary['chunks']} | skipped: {summary['skipped']} | "
                    f"parsed: {summary['parsed']} files, cached: {summary['cached']}"
                )
                status.update(label="Ingest complete", state="complete")
                get_resources.clear()
            except Exception as exc:
                status.update(label="Ingest failed", state="error")
                st.error(str(exc))

    if st.button("Reset DB + re-ingest", use_container_width=True):
        with st.status("Resetting database...", expanded=True) as status:
            try:
                chain.get_vectorstore(check_embedding=False).reset_collection()
                get_resources.clear()
                summary = ingest.ingest()
                st.write(
                    f"Files: {summary['files']} | chunks added: "
                    f"{summary['chunks']} | skipped: {summary['skipped']} | "
                    f"parsed: {summary['parsed']} files, cached: {summary['cached']}"
                )
                status.update(label="Reset and re-ingest complete", state="complete")
                get_resources.clear()
            except Exception as exc:
                status.update(label="Reset failed", state="error")
                st.error(str(exc))

    k = st.slider("Retriever k", min_value=1, max_value=8, value=4)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    with st.expander("Quick tests"):
        for i, question in enumerate(QUESTIONS):
            if st.button(question, key=f"quick_{i}", use_container_width=True):
                st.session_state.pending = question
                st.rerun()

store, answer_chain, has_docs = get_resources(k)

if not has_docs:
    st.warning(
        "Vector DB is empty. Add documents to data/ and click 'Ingest papers/'."
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_meta(
                msg.get("sources", []),
                msg.get("latency", 0.0),
                msg.get("usage", {}),
            )

if st.session_state.pending:
    prompt = st.session_state.pending
    st.session_state.pending = None
else:
    prompt = st.chat_input("Ask a question...", disabled=not has_docs)

if prompt:
    history = [
        HumanMessage(content=m["content"])
        if m["role"] == "user"
        else AIMessage(content=m["content"])
        for m in st.session_state.messages[-HISTORY_TURNS * 2 :]
    ]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving..."):
                results = store.similarity_search_with_score(prompt, k=k)
            context = chain.format_docs([doc for doc, _ in results])
            sources = [
                {
                    "name": Path(doc.metadata.get("source", "unknown")).name,
                    "text": doc.page_content,
                    "score": float(score),
                }
                for doc, score in results
            ]
            usage: dict = {}

            def stream():
                payload = {
                    "context": context,
                    "chat_history": history,
                    "question": prompt,
                }
                for attempt in range(4):
                    try:
                        for chunk in answer_chain.stream(payload):
                            if getattr(chunk, "usage_metadata", None):
                                usage.update(chunk.usage_metadata)
                            if chunk.content:
                                yield chunk.content
                        return
                    except Exception as exc:
                        if not _is_429(exc) or attempt == 3:
                            raise
                        time.sleep(2**attempt)

            start = time.perf_counter()
            answer = st.write_stream(stream())
            latency = time.perf_counter() - start

            render_meta(sources, latency, usage)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "latency": latency,
                    "usage": usage,
                }
            )
        except Exception as exc:
            code = _status_code(exc)
            if _is_429(exc):
                st.error("Cohere trial rate/usage limit reached")
            elif code in (401, 404):
                st.error("invalid key or model name")
            else:
                st.error(f"API error: {exc}")
