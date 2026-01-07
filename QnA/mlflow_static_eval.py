import os
import mlflow
import pandas as pd

from dotenv import load_dotenv
from openai import AzureOpenAI

from mlflow.metrics.genai import (
    answer_relevance,
    answer_correctness, 
    answer_similarity
)

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

    # Configurar nuevas variables de entorno 
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["OPENAI_API_VERSION"] = openai_api_version
    os.environ["OPENAI_API_TYPE"] = openai_api_type or "azure"
    os.environ["OPENAI_API_BASE"] = openai_api_base
    os.environ["OPENAI_DEPLOYMENT_NAME"] = openai_deployment_name

    # Configurar el cliente de Azure OpenAI
    client = AzureOpenAI(
        azure_endpoint=openai_api_base,
        api_key=openai_api_key,
        api_version=openai_api_version
    )

    return client, openai_deployment_name


def generate_answer(question):
    system_prompt = """
    Eres un asistente útil que responde preguntas en español de manera concisa y precisa.
    """
    client, openai_deployment_name = load_mlflow_nedded_resources()
    response = client.chat.completions.create(
        model=openai_deployment_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Pregunta: {question}"}
        ],
        temperature=0.0,
        top_p=1.0
    )
    answer = response.choices[0].message.content.strip()
    return {"answer": answer}

def generate_agent_responses(df):
    df_res = df[:3].copy()
    df_res["predictions"] = ""

    for i, row in df_res.iterrows():
        try:
            response = generate_answer(row["inputs"])
            df_res.at[i, "predictions"] = response["answer"]
        except Exception as e:
            print(str(e))
            df_res.at[i, "predictions"] = f"Error: {e}"

    df_res.to_excel("df_static_QnA_eval.xlsx", index=False)

    return df_res


# Configurar la URI de seguimiento de MLflow
mlflow.set_tracking_uri("http://localhost:8080")
# Configurar el experimento de MLflow
mlflow.set_experiment("eval_openai_gpt4_spanish")

# Run de evaluación en MLflow
with mlflow.start_run(run_name="llm_QnA_estatic_evaluate") as run:    
    print(f"Starting MLflow run: {run.info.run_id}")
    client, openai_deployment_name = load_mlflow_nedded_resources()
    
    # Ejemplo de conjunto de datos de evaluación 
    eval_ground_truth_data = [
        {
            "inputs": "¿Cuál es la capital de Francia?",
            "targets": "La capital de Francia es París.",
        },
        {
            "inputs": "¿Qué es la inteligencia artificial?",
            "targets": "La inteligencia artificial es un campo de la informática que busca crear sistemas capaces de realizar tareas que requieren inteligencia humana.",
        },
        {
            "inputs": "¿Cuándo se fundó Microsoft?",
            "targets": "Microsoft fue fundada en 1975 por Bill Gates y Paul Allen.",
        }
    ]

    eval_ground_truth_df = pd.DataFrame({
        "inputs": [item["inputs"] for item in eval_ground_truth_data],
        "targets": [item["targets"] for item in eval_ground_truth_data]
    })
    
    # Generar respuestas del agente para el conjunto de datos de evaluación
    eval_df = generate_agent_responses(eval_ground_truth_df)

    # Evaluación estática (con predicciones realizadas previamente)
    results = mlflow.evaluate(
        data=eval_df,
        targets="targets",
        predictions="predictions",
        model_type="question-answering",
        extra_metrics=[    
            answer_similarity(),
            answer_relevance(),
            answer_correctness()
        ]
    )
    print(f"See aggregated evaluation results below: \n{results.metrics}")

    eval_table = results.tables["eval_results_table"]
    print(type(eval_table))
    print(f"See evaluation table below: \n{eval_table}")

    df_eval_table = pd.DataFrame(eval_table)
    df_eval_table.to_excel("static_QnA_evaluation_results.xlsx", index=False)
