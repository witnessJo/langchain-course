"""LangChain Course package."""

from .config import CHROMA_PERSIST_DIR, ORG_ROAM_DIR, PROJECT_ROOT


def main() -> None:
    from .ingestion import ingest_org_roam
    from .qa import answer_question

    added, removed = ingest_org_roam()
    print(f"Org-roam ingestion complete: {added} chunks indexed, {removed} removed")

    while True:
        try:
            question = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in {"exit", "quit", ":q"}:
            break
        if not question:
            continue

        answer, sources = answer_question(
            question, on_progress=lambda message: print(message, flush=True)
        )
        print(f"\n{answer}")
        if sources:
            print("\n검색된 노트:")
            for index, source in enumerate(sources, start=1):
                print(f"[{index}] {source}")


__all__ = ["ORG_ROAM_DIR", "CHROMA_PERSIST_DIR", "PROJECT_ROOT", "main"]
__version__ = "0.1.0"
