import asyncio
import os
import ssl
import certifi

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from typing import Any, Dict, List

from logger import Colors, log_error, log_header, log_info, log_success, log_warning

load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


def chunk_urls(urls: list[str], chunk_size: int = 20) -> list[list[str]]:
    """Split URLs into chunks of specified size."""
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i : i + chunk_size]
        chunks.append(chunk)
    return chunks


async def extract_batch(urls: List[str], batch_num: int) -> dict[str, Any]:
    """Extract document from batch of URLs."""
    try:
        log_info(
            f"TavilyExtract: Processing batch {batch_num} with {len(urls)} URLs",
            Colors.BLUE,
        )
        docs = await tavily_extract.ainvoke(input={"urls": urls})
        log_success(
            f"TavilyExtract: Completed batch {batch_num} - extracted {len(docs.get('results', []))} documents"
        )
        return docs
    except Exception as e:
        log_error(f"TavilyExtract: Failed to extact batch {batch_num} - {e}")
        return {"results": []}


async def async_extract(url_batches: list[list[str]]):
    log_header("DOCUMENT EXTRACTION PHASE")
    log_info(
        f"\ TavilyExtract: Starting concurrent extraction of {len(url_batches)} batches",
        Colors.DARKCYAN,
    )

    tasks = [extract_batch(batch, i + 1) for i, batch in enumerate(url_batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out excetions and flatten result
    all_pages = []
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            log_error(f"TavilyExtract Batch failed with exception - {result}")
            failed_batches += 1
        else:
            for extracted_page in result["results"]:  # ty: ignore[not-subscriptable]
                if not extracted_page.get("raw_content"):
                    continue
                document = Document(
                    page_content=extracted_page["raw_content"],
                    metadata={"source": extracted_page["url"]},
                )
                all_pages.append(document)

    log_success(
        f"TavilyExtract: Extraction complete! Total pages extraced: {len(all_pages)}"
    )
    if failed_batches > 0:
        log_warning(f"TavilyExtract: {failed_batches} batches failed during extraction")

    return all_pages


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PILELINE")
    log_info(
        "** TavilyCrawl: Starting to Crawl documentation from https://python.langchain.com/",
        Colors.PULPLE,
    )

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 5,
            "extract_depth": "advanced",
        }
    )
    print(res)
    all_docs = [
        Document(page_content=result["raw_content"], metadata={"source": result["url"]})
        for result in res["results"]
        if result.get("raw_content")
    ]
    site_map = tavily_map.invoke("https://python.langchain.com/")
    log_success(
        f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documentation site"
    )

    url_batches = chunk_urls(list(site_map["results"]), chunk_size=20)
    log_info(
        f"URL Processing: Split {len(site_map['results'])} URLs info {len(url_batches)} batches"
    )

    # Extract documents from URLs
    all_docs = await async_extract(url_batches)
    print(all_docs)


if __name__ == "__main__":
    asyncio.run(main())
