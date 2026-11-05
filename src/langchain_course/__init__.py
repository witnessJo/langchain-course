"""LangChain Course package."""

from .config import CHROMA_PERSIST_DIR, ORG_ROAM_DIR, PROJECT_ROOT


def main() -> None:
    from .ingestion import ingest_org_roam

    added, removed = ingest_org_roam()
    print(f"Org-roam ingestion complete: {added} chunks indexed, {removed} removed")


__all__ = ["ORG_ROAM_DIR", "CHROMA_PERSIST_DIR", "PROJECT_ROOT", "main"]
__version__ = "0.1.0"
