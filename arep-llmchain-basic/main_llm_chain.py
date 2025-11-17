import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1) Cargar variables de entorno
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("Falta la variable OPENAI_API_KEY en el archivo .env")

# 2) Crear el LLM de LangChain
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
)

# 3) Crear el prompt template
prompt = ChatPromptTemplate.from_template(
    "Eres un profesor de IA. Explica en 3 frases y en lenguaje sencillo el concepto de: {topic}"
)

# 4) Construir la cadena (LCEL)
chain = prompt | llm | StrOutputParser()


def main():
    topic = input("Tema a explicar: ")
    result = chain.invoke({"topic": topic})

    print("\n=== Respuesta del modelo ===\n")
    print(result)


if __name__ == "__main__":
    main()
