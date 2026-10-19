from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

load_dotenv()

if __name__ == "__main__":
    print("Loading...")
    documents = UnstructuredLoader("mediumblog1.txt").load()

    print("Splitting...")
    chunks = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0).split_documents(
        documents
    )
    print(f"Created {len(chunks)} chunks")

    print("Ingesting...")
    Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        collection_name="medium-blogs",
        persist_directory="./chroma_db",
    )
    print(f"Ingested {len(chunks)} chunks")
