import os
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from tqdm.auto import tqdm
from datasets import load_dataset
from dotenv import load_dotenv


load_dotenv()

# === 0. Configuración de Azure OpenAI ===
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_API_TYPE = os.getenv("AZURE_OPENAI_API_TYPE")
OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")

# === 1. Configuración de modelo de Embeddings para DB Vectorial ===
embedding = AzureOpenAIEmbeddings(
    model=OPENAI_EMBEDDING_MODEL,
    azure_endpoint=AZURE_OPENAI_API_BASE,
    api_key=AZURE_OPENAI_API_KEY, # sin el open_ para esta version de langchain-openai
    api_version=AZURE_OPENAI_API_VERSION, # sin el open_ para esta version de langchain-openai
    chunk_size=1000
)

# === 2. Creación de DB Vectorial para después cargar ===
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)

# === 3. Limpieza y Adición de textos y metadatos a la DB Vectorial
def clean_special_tokens(text):
    """
    Esta función remueve el token especial <|endoftext|>, que a veces aparece al 
    final de ciertos textos generados por modelos de lenguaje. Este token no aporta información 
    semántica útil y puede interferir en el proceso de embedding o recuperación si se deja tal cual.
    """
    return text.replace("<|endoftext|>", " ").strip()

dataset = load_dataset("jamescalam/ai-arxiv2-chunks", split="train[:20000]")
data = dataset.to_pandas()

# Obtener textos ya cargados
existing_docs = vectorstore.get()
existing_texts = set(existing_docs["documents"])

batch_size = 100
for i in tqdm(range(0, len(data), batch_size)):
    i_end = min(len(data), i + batch_size)
    batch = data.iloc[i:i_end]

    texts = batch["chunk"].tolist()
    metadatas = batch[["source", "title", "chunk"]].to_dict(orient="records")

    # Filtrar textos que no están ya en la base y limpiar los nuevos
    new_texts_and_metas = [
        (clean_special_tokens(t), m) for t, m in zip(texts, metadatas) if t not in existing_texts
    ]

    if new_texts_and_metas:
        # Añadir textos y sus metadatos 
        new_texts, new_metas = zip(*new_texts_and_metas)
        vectorstore.add_texts(texts=list(new_texts), metadatas=list(new_metas))