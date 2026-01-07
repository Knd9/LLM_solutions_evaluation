import asyncio
import os
import mlflow
import pandas as pd

from dotenv import load_dotenv
from datasets import load_dataset

from mlflow.metrics.genai import (
    answer_relevance,
    answer_similarity,
    answer_correctness,
    relevance,
    faithfulness,
)
from mlflow.models import make_metric

from ragas.metrics import FactualCorrectness, SemanticSimilarity
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from dummy_agent.src.agent_langchain import generate_answer_and_context


def load_mlflow_nedded_resources():
    """
    Cargar las variables de entorno desde el archivo .env.
    Luego asignar las que espera MLFlow para cargar un modelo de openai.
    Para MLFlow, tener AZURE_OPENAI_ENDPOINT asignada da error,
    porque son mutuamente exclusivas con OPENAI_API_BASE.
    Para Langchain, tener OPENAI_API_BASE asignada da error, 
    porque son mutuamente exclusivas con AZURE_OPENAI_API_BASE.
    Por estos motivos, utilizar esta función luego de generar las 
    respuestas del agente, al menos al usar langchain/langgraph.
    """
    # Cargar las variables de entorno
    load_dotenv()

    openai_api_base = os.getenv("AZURE_OPENAI_API_BASE")
    openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    openai_api_type = os.getenv("AZURE_OPENAI_API_TYPE")
    openai_deployment_name = os.getenv("OPENAI_DEPLOYMENT_NAME")
    openai_embeddings_model = os.getenv("OPENAI_EMBEDDING_MODEL")

    # Configurar el cliente de Azure OpenAI y Azure Embeddings 
    llm_client = AzureChatOpenAI(
        azure_deployment=openai_deployment_name,
        api_version=openai_api_version,
        api_key=openai_api_key,
        azure_endpoint=openai_api_base,
    )
    embeddings_client = AzureOpenAIEmbeddings(
        api_version=openai_api_version,
        azure_endpoint=openai_api_base,
        api_key=openai_api_key,
        model=openai_embeddings_model
    )

    # Configurar nuevas variables de entorno 
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["OPENAI_API_VERSION"] = openai_api_version
    os.environ["OPENAI_API_TYPE"] = openai_api_type or "azure"
    os.environ["OPENAI_API_BASE"] = openai_api_base
    os.environ["OPENAI_DEPLOYMENT_NAME"] = openai_deployment_name

    return llm_client, openai_deployment_name, embeddings_client


def generate_agent_responses(df):
    """Generar respuestas del sistema agente para las preguntas del DataFrame."""
    df_res = df[:3].copy()
    df_res["answer"] = ""
    df_res["contexts"] = None

    for i, row in df_res.iterrows():
        try:
            response = generate_answer_and_context(row["question"])
            df_res.at[i, "answer"] = response["answer"]
            df_res.at[i, "contexts"] = response.get("contexts", [])
        except Exception as e:
            df_res.at[i, "answer"] = f"Error: {e}"
            df_res.at[i, "contexts"] = []

    df_res.to_excel("df_eval.xlsx", index=False)

    return df_res


def ensure_list(x):
    """Asegurarse de que x sea una lista."""
    if isinstance(x, list):
        return x
    elif pd.isna(x):
        return []
    else:
        return [x]


def to_mlflow_dataset(df):
    """Renombrar columnas del DataFrame a las esperadas por MLFlow."""
    df = df.copy()
    df.rename(columns={"question": "inputs"}, inplace=True)
    df.rename(columns={"ground_truth": "targets"}, inplace=True)

    if "answer" in df.columns:
        df.rename(columns={"answer": "predictions"}, inplace=True)

    if "contexts" in df.columns:
        # Elegir contexts o reference_contexts según enfoque (métrica electa)
        df["context"] = df["contexts"].apply(ensure_list)
        #df.rename(columns={"reference_contexts": "context"}, inplace=True)

    interest_columns = ['inputs', 'targets', 'context', 'predictions']

    df_res = df[interest_columns]

    df_res.to_excel("df_mlflow_eval.xlsx", index=False)

    return df_res

# Ragas metrics
def factual_correctness_scorer(predictions, targets):        
    scores = [
        asyncio.run(
            FactualCorrectness(llm=evaluator_llm).single_turn_ascore(
                sample=SingleTurnSample(response=pred, reference=targ)
            )
        )
        for pred, targ in zip(predictions.astype(str).values.tolist(), 
                                targets.astype(str).values.tolist())
    ]

    avg_score = sum(scores) / len(scores)
    print("Factual_Correctness= ", avg_score)
    return avg_score


def semantic_similarity_scorer(predictions, targets):
    scores = [
        asyncio.run(
            SemanticSimilarity(embeddings=evaluator_emb).single_turn_ascore(
                sample=SingleTurnSample(response=pred, reference=targ)
            )
        )
        for pred, targ in zip(predictions.astype(str).values.tolist(), 
                                targets.astype(str).values.tolist())
    ]

    avg_score = sum(scores) / len(scores)
    print("Semantic_Similarity= ", avg_score)
    return avg_score

# Configurar URL de tracking de MLFlow
mlflow.set_tracking_uri("http://localhost:8080")
# Configurar experimento
mlflow.set_experiment("eval_azopenai_rag_langgraph")
# Crear e iniciar run
with mlflow.start_run(run_name="llm_rag_evaluate") as run:
    print(f"Starting MLflow run: {run.info.run_id}")
    # Asignamos variables para cargar modelo en MLFlow
    llm_client, _, embeddings_client = load_mlflow_nedded_resources()
    evaluator_llm = LangchainLLMWrapper(llm_client)
    evaluator_emb = LangchainEmbeddingsWrapper(embeddings_client)

    # Cargar dataset
    eval_ground_truth_data = load_dataset("aurelio-ai/ai-arxiv2-ragas-mixtral", split="train")
    eval_ground_truth_df = eval_ground_truth_data.to_pandas()
    # Generar predicciones
    df = generate_agent_responses(eval_ground_truth_df)
    # Renombrar columnas a las esperadas por MLFlow para evitar pasar todas en evaluate
    eval_df = to_mlflow_dataset(df)
    print(eval_df.columns)

    # Ragas metrics
    factual_correctness_metric = make_metric(
        eval_fn= factual_correctness_scorer,
        name= "ragas_factual_correctness",
        greater_is_better= True
    )
    semantic_similarity_metric = make_metric(
        eval_fn= semantic_similarity_scorer,
        name= "ragas_semantic_similarity",
        greater_is_better= True
   )

    # Evaluar con dataset estático (con predicciones incluídas)
    results = mlflow.evaluate(
        data=eval_df,
        targets="targets",
        predictions="predictions",
        model_type="text",
        extra_metrics=[
            factual_correctness_metric,
            semantic_similarity_metric,
            answer_similarity(),
            answer_relevance(),
            answer_correctness(),
            relevance(),
            faithfulness()
        ]
    )
    print(f"See aggregated evaluation results below: \n{results.metrics}") 

    eval_table = results.tables["eval_results_table"]
    df_eval_table = pd.DataFrame(eval_table)
    df_eval_table.to_excel("evaluation_results.xlsx", index=False)
