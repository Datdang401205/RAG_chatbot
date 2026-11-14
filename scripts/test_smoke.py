"""Smoke test: two RAG questions with a Cohere call budget."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chain import build_rag_chain

QUESTIONS = ["Who is the CISO?", "Who is the CTO of Zentora Labs?"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-calls", type=int, default=30)
    args = parser.parse_args()

    chain = build_rag_chain()
    calls = 0
    for index, question in enumerate(QUESTIONS):
        if calls >= args.max_calls:
            print(f"STOP: max calls ({args.max_calls}) reached")
            return 1
        calls += 1
        answer = chain.invoke({"question": question, "chat_history": []})
        print(f"Q: {question}\nA: {answer}\n")
        if index < len(QUESTIONS) - 1:
            time.sleep(1)

    print(f"Cohere calls: {calls}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
