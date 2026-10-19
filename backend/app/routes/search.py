import os
from fastapi import APIRouter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

router = APIRouter(prefix="/api")


# Create the embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
)


# Connect to our existing ChromaDB
vector_db = Chroma(
    collection_name=os.environ.get("CHROMA_COLLECTION", "akos_documents_multilingual"),
    embedding_function=embedding_model,
    persist_directory=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "chroma_db")
)


@router.get("/search")
async def search_documents(query: str):

    results = vector_db.similarity_search(
        query,
        k=3
    )

    documents = []

    for result in results:
        documents.append(result.page_content)

    return {
        "query": query,
        "results": documents
    }
