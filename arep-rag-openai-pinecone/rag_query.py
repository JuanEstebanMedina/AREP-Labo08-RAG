import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def main():
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not openai_key or not pinecone_key or not index_name:
        raise RuntimeError("Revisa que OPENAI_API_KEY, PINECONE_API_KEY y PINECONE_INDEX_NAME estén en el .env")

    # 1) Embeddings y vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 2) Modelo de lenguaje
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
    )

    # 3) Prompt para RAG
    template = """
Eres un asistente que responde únicamente con base en el siguiente contexto.
Si la respuesta no está en el contexto, di claramente que no aparece en los documentos.

Contexto:
{context}

Pregunta: {question}

Respuesta:
"""
    prompt = ChatPromptTemplate.from_template(template)

    # 4) Cadena RAG con LCEL
    # - "question" pasa directo
    # - "context" se obtiene llamando al retriever con la misma pregunta
    rag_chain = (
        {
            "question": RunnablePassthrough(),
            "context": retriever,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("Asistente RAG listo. Escribe 'salir' para terminar.\n")

    while True:
        question = input("Pregunta: ")
        if question.lower() in ("salir", "exit", "q"):
            break

        answer = rag_chain.invoke(question)
        print("\n--- Respuesta ---\n")
        print(answer)
        print("\n-----------------\n")


if __name__ == "__main__":
    main()
