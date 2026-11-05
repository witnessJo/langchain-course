"""Index Org-roam notes in a dedicated Chroma collection."""

import hashlib

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHROMA_PERSIST_DIR, ORG_ROAM_DIR

COLLECTION_NAME = "org_roam"
BATCH_SIZE = 100


def ingest_org_roam() -> tuple[int, int]:
    if not ORG_ROAM_DIR.is_dir():
        raise FileNotFoundError(f"Org-roam directory does not exist: {ORG_ROAM_DIR}")

    paths = sorted(ORG_ROAM_DIR.rglob("*.org"))
    if not paths:
        raise FileNotFoundError(f"No .org files found in {ORG_ROAM_DIR}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    documents: dict[str, Document] = {}
    for path in paths:
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            continue

        relative_path = path.relative_to(ORG_ROAM_DIR).as_posix()
        source = str(path.resolve())
        for index, chunk in enumerate(splitter.split_text(content)):
            chunk_id = hashlib.sha256(f"{relative_path}:{index}".encode()).hexdigest()
            documents[chunk_id] = Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "ingestion_source": COLLECTION_NAME,
                    "content_hash": hashlib.sha256(chunk.encode()).hexdigest(),
                },
            )

    if not documents:
        raise ValueError(f"No nonempty .org files found in {ORG_ROAM_DIR}")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    existing = vectorstore.get(include=["metadatas"])
    existing_metadata = dict(zip(existing["ids"], existing["metadatas"]))
    changed_ids = [
        chunk_id
        for chunk_id, document in documents.items()
        if existing_metadata.get(chunk_id) != document.metadata
    ]

    for offset in range(0, len(changed_ids), BATCH_SIZE):
        batch_ids = changed_ids[offset : offset + BATCH_SIZE]
        vectorstore.add_documents(
            [documents[chunk_id] for chunk_id in batch_ids], ids=batch_ids
        )

    stale_ids = list(existing_metadata.keys() - documents.keys())
    for offset in range(0, len(stale_ids), BATCH_SIZE):
        vectorstore.delete(ids=stale_ids[offset : offset + BATCH_SIZE])

    return len(changed_ids), len(stale_ids)
