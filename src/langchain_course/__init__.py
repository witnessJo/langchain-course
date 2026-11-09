"""LangChain Course package."""

try:
    import readline  # noqa: F401
except ImportError:
    pass

from .config import CHROMA_PERSIST_DIR, ORG_ROAM_DIR, PROJECT_ROOT


def main() -> None:
    from .ingestion import ingest_org_roam
    from .qa import answer_question

    print("[*] Org-roam 노트 색인 확인 중...", flush=True)
    added, removed = ingest_org_roam()
    print(f"[*] 색인 완료: {added}개 청크 추가, {removed}개 삭제됨\n", flush=True)

    while True:
        try:
            question = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in {"exit", "quit", ":q"}:
            break
        if not question:
            continue

        print("\n[*] 질문을 전송했습니다. 분석 중...", flush=True)

        def show_progress(message: str) -> None:
            print(f"[*] {message}", flush=True)

        token_streamed = False

        def show_token(token: str) -> None:
            nonlocal token_streamed
            if not token_streamed:
                print("\n[답변]", flush=True)
                token_streamed = True
            print(token, end="", flush=True)

        answer, sources = answer_question(
            question,
            on_progress=show_progress,
            on_token=show_token,
        )

        if not token_streamed:
            print(f"\n[답변]\n{answer}")
        else:
            print()

        if sources:
            print("\n검색된 노트:")
            for index, source in enumerate(sources, start=1):
                print(f"[{index}] {source}")
        print("-" * 50)


__all__ = ["ORG_ROAM_DIR", "CHROMA_PERSIST_DIR", "PROJECT_ROOT", "main"]
__version__ = "0.1.0"
