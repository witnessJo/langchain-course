import asyncio
import os
import ssl

import certifi

# WebBaseLoader asks for this so the site can see who is crawling it, and it
# checks at import time -- hence before langchain_community comes in below.
os.environ.setdefault("USER_AGENT", "langchain-course-ingestion/0.1")

from typing import List

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import SitemapLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger import (
    Colors,
    log_error,
    log_header,
    log_info,
    log_success,
    log_warning,
)

load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# python.langchain.com answers 308 and sends everything to docs.langchain.com,
# whose sitemap lists the whole documentation set for free.  Asking a search
# API to rediscover these URLs, and then to extract each page, is what ran the
# Tavily quota out.
SITEMAP_URL = "https://docs.langchain.com/sitemap.xml"
URL_FILTERS = [r"https://docs\.langchain\.com/oss/python/.*"]
REQUESTS_PER_SECOND = 2

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)


def parse_documentation_page(soup: BeautifulSoup) -> str:
    """Keep a documentation page's body and drop the furniture around it."""
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    # Syntax highlighting splits code into spans, which a newline separator
    # would spread over one line each.  Flatten them back first.
    for pre in soup.find_all("pre"):
        pre.string = pre.get_text(separator="", strip=False)
    main = soup.find("main") or soup.find("article") or soup
    return main.get_text(separator="\n", strip=True)


def load_documentation() -> List[Document]:
    """Read every documentation page the sitemap lists."""
    loader = SitemapLoader(
        web_path=SITEMAP_URL,
        filter_urls=URL_FILTERS,
        parsing_function=parse_documentation_page,
        continue_on_failure=True,
        requests_per_second=REQUESTS_PER_SECOND,
    )
    return loader.load()


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PILELINE")
    log_info(
        f"** SitemapLoader: Reading documentation URLs from {SITEMAP_URL}",
        Colors.PULPLE,
    )

    try:
        # `load' is blocking, so keep it off this coroutine's thread.
        all_docs = await asyncio.to_thread(load_documentation)
    except Exception as e:
        log_error(f"SitemapLoader: Failed to load documentation - {e}")
        return

    empty_docs = [doc for doc in all_docs if not doc.page_content.strip()]
    all_docs = [doc for doc in all_docs if doc.page_content.strip()]
    if empty_docs:
        log_warning(f"SitemapLoader: Skipped {len(empty_docs)} pages with no content")

    if not all_docs:
        log_error("SitemapLoader: No documents were loaded, nothing left to do")
        return

    log_success(
        f"SitemapLoader: Successfully loaded {len(all_docs)} pages from documentation site"
    )

    # Split documents info chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )


if __name__ == "__main__":
    asyncio.run(main())
