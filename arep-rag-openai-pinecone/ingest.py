import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore


def load_documents(data_path: str = "data"):
    """Carga todos los .txt de la carpeta data/ y los devuelve como documentos de LangChain."""
    docs = []
    for file_path in Path(data_path).glob("*.txt"):
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs.extend(loader.load())
    return docs


def split_documents(documents):
    """Divide documentos grandes en chunks más pequeños."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    return splitter.split_documents(documents)


def main():
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not openai_key:
        raise RuntimeError("Falta OPENAI_API_KEY en el .env")
    if not pinecone_key:
        raise RuntimeError("Falta PINECONE_API_KEY en el .env")
    if not index_name:
        raise RuntimeError("Falta PINECONE_INDEX_NAME en el .env")

    print("Cargando documentos...")
    docs = load_documents()
    print(f"Documentos cargados: {len(docs)}")

    print("Dividiendo en chunks...")
    chunks = split_documents(docs)
    print(f"Chunks generados: {len(chunks)}")

    print("Creando embeddings y subiendo a Pinecone...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # PineconeVectorStore.from_documents creará o usará el índice indicado
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        # pinecone_api_key se toma automáticamente de PINECONE_API_KEY
    )

    print("Indexación terminada.")
    print(f"Vector store listo en el índice: {index_name}")


if __name__ == "__main__":
    main()
