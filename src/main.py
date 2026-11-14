"""CLI chat loop with short multi-turn history and streaming output."""

import sys

from langchain_core.messages import AIMessage, HumanMessage

from src.chain import build_rag_chain
from src.config import HISTORY_TURNS, require_api_key


def main() -> None:
    """Run the interactive chat loop until the user types 'exit'."""
    try:
        require_api_key()
        chain = build_rag_chain()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    history: list = []
    print("RAG chatbot ready. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() == "exit":
            break

        print("Assistant: ", end="", flush=True)
        chunks: list[str] = []
        try:
            for chunk in chain.stream(
                {"question": question, "chat_history": history}
            ):
                print(chunk, end="", flush=True)
                chunks.append(chunk)
        except Exception as exc:
            print(f"\n[Error] {exc}")
            continue

        answer = "".join(chunks)
        print("\n")
        history.extend(
            [HumanMessage(content=question), AIMessage(content=answer)]
        )
        history = history[-HISTORY_TURNS * 2 :]


if __name__ == "__main__":
    main()
