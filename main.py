import pprint
from unstructured.partition.pdf_image.pdfminer_processing import (
    check_annotations_within_element,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

import os
from operator import itemgetter

load_dotenv()

print("Initializing components...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()
vectorstore = Chroma(
    collection_name="medium-blogs",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the follwing context:

{context}

Question: {question}

Provide a detailed answer:"""
)


def create_retrieval_chian_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Return a chian that can be invoked with {"question": "..."}
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    query = "What is Pinecone in machine learning?"
    chain_with_lcel = create_retrieval_chian_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)


if __name__ == "__main__":
    main()
