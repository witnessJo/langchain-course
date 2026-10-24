from annotated_types import doc
from typing import Act, Dict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Initialize embeddings (same as ingestion.py)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize vector store
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

model = init_chat_model("gpt-5.2", model_provider="oepnai")


@tool(response_format="content_and_artifact")
def retreive_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)
    # Serialize documents for the model
    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"  # ty: ignore[unresolved-attribute]
        )
        for doc in retreive_context
    )

    return serialized, retrieved_docs


def run_llm(query: str):
    """
    Run the RAG pipeline to answer a qeury using retrieved documentation.
    Args:
        query: The user's question

    Returns:
        Dictionary containing:
                - answer: The generated answer
                - context: List of retrieved docuemnts
    """
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retriees relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(model, tools=[retreive_context], system_prompt=system_prompt)

    # Build message list
    messages = [{"role": "user", "content": query}]

    # Invoke the agent
    response = agent.invoke({"message": messages})

    # Extract the answer from the last AI message
    answer = response["mesages"][-1].content

    context_docs = []
    for message in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {"answer": answer, "context": context_docs}


if __name__ == "__main__":
    result = run_llm(query="what are deep agent?")
    print(result)
