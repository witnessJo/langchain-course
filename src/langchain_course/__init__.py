"""LangChain Course package."""

from .config import CHROMA_PERSIST_DIR, ORG_ROAM_DIR, PROJECT_ROOT


def main() -> None:
    print("Hello from langchain-course!")
    print(f"ORG_ROAM_DIR: {ORG_ROAM_DIR}")
    print(f"CHROMA_PERSIST_DIR: {CHROMA_PERSIST_DIR}")


__all__ = ["ORG_ROAM_DIR", "CHROMA_PERSIST_DIR", "PROJECT_ROOT", "main"]
__version__ = "0.1.0"
