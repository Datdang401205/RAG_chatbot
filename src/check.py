"""Startup check: Cohere chat and local embeddings."""

import sys

from src.config import COHERE_MODEL, EMBED_MODEL, get_embeddings, get_llm


def _status(exc: Exception) -> int | None:
    return (
        getattr(exc, "status_code", None)
        or getattr(getattr(exc, "response", None), "status_code", None)
        or getattr(exc, "status", None)
    )


def main() -> int:
    """Print one PASS/FAIL line for chat and embeddings."""
    try:
        reply = get_llm().invoke("hi", max_tokens=5)
        usage = getattr(reply, "usage_metadata", None) or {}
        print(
            f"PASS chat (200, {COHERE_MODEL}, "
            f"output_tokens={usage.get('output_tokens', 0)})"
        )
    except Exception as exc:
        code = _status(exc)
        print(f"FAIL chat ({code or 'error'}, {exc})")
        if code in (401, 404) or any(s in str(exc) for s in ("401", "404")):
            print("STOP: check COHERE_API_KEY and COHERE_MODEL.")
            return 1

    try:
        vector = get_embeddings().embed_query("ping")
        print(f"PASS embeddings (200, {EMBED_MODEL}, dim={len(vector)})")
    except Exception as exc:
        code = _status(exc)
        print(f"FAIL embeddings ({code or 'error'}, {exc})")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
