import os
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from logging import getLogger
from dotenv import load_dotenv


load_dotenv()
logger = getLogger(__name__)
# === 0. Configuración de Azure OpenAI ===
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_API_TYPE = os.getenv("AZURE_OPENAI_API_TYPE")
OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")

# === 1. Configuración de modelo de Embeddings para DB Vectorial ===
# embedding = AzureOpenAIEmbeddings(
#     model=OPENAI_EMBEDDING_MODEL,
#     azure_endpoint=AZURE_OPENAI_API_BASE,
#     api_key=AZURE_OPENAI_API_KEY, # sin el openai_ para la version langchain-openai==0.3.28
#     api_version=AZURE_OPENAI_API_VERSION, # sin el openai_ para la version langchain-openai==0.3.28
#     chunk_size=1000
# )
embeddings_client = AzureOpenAIEmbeddings(
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_API_BASE,
    api_key=AZURE_OPENAI_API_KEY,
    model=OPENAI_EMBEDDING_MODEL,
    chunk_size=1000
)

# === 2. Cargar la DB Vectorial (en este caso ya creada) ===
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings_client)

# === 3. Crear Herramienta para buscar en Chroma ===
@tool
def arxiv_search(query: str) -> str:
    """Usa esta tool para responder preguntas sobre IA, aprendizaje automático, 
    ciencia de datos u otras preguntas técnicas que puedan responderse mediante 
    artículos de arXiv."""
    try:
        results = vectorstore.similarity_search(query, k=5)
        logger.info("[DEBUG] Chroma search results:")
        for i, doc in enumerate(results):
            logger.info(f"Chunk {i+1}: {doc.page_content[:200]}...")
        return "\n---\n".join([doc.page_content for doc in results])
    except Exception as e:
        logger.error(f"Error en búsqueda: {e}")
        return ""


tools = [arxiv_search]

# === 4. LLM + LangGraph agent ===

llm_client = AzureChatOpenAI(
    azure_deployment=OPENAI_DEPLOYMENT_NAME,
    api_version=AZURE_OPENAI_API_VERSION,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_API_BASE,
    temperature=0.0,
    top_p=1.0
)
# Crear agente LangGraph ReAct
langgraph_agent_executor = create_react_agent(model=llm_client, tools=tools)#, debug=True)

# === 5. Función de inferencia expuesta por la API ===
def generate_answer_and_context(question: str) -> dict:
    messages = langgraph_agent_executor.invoke({
        "messages": [
                ("system", "Allways answer the question using the provided Tool."),
                ("human", question)
            ]
        }
    )
    output = messages["messages"][-1].content
    contexts = [
        m.content for m in messages["messages"]
        if getattr(m, "name", None) == "arxiv_search"
    ]
    if contexts:
        logger.info("[DEBUG] Contexts returned:")
        for i, ctx in enumerate(contexts):
            logger.info(f"Context {i+1}: {ctx[:200]}...")
    return {
        "answer": output,
        "contexts": contexts
    }