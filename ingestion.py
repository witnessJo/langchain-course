import asyncio
import os
import ssl
import certifi

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import Any, Dict, List


async def main():
    """Main async function to orchestrate the entire process."""


if __name__ == "__main__":
    asyncio.run(main())
