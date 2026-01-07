# MLFlow Evaluation

Evaluating Data & AI Use Cases with MLFlow

## Template Structure

- `README.md`: Step-by-step guide to run the evaluation.
- `.env_template`: Environment variables required for configuration. A copy must be placed and renamed to .env inside the folder corresponding to the use case of interest.
- `Classification/`: Evaluation for this use case.

  - `classification_metrics_explain.png`: Spanish explanation of classification metrics in a confusion matrix.
  - `constants.py`: Application entry point.
  - `mlflow_eval.py`: Evaluation script to be executed.
  - `requirements.txt`: Project dependencies.

- `QnA/`: Evaluation for this use case.

  - `mlflow_dynamic_eval.py`: Evaluation script to be executed dynamically (without precomputed predictions).
  - `mlflow_static_eval.py`: Evaluation script to be executed statically (with previously generated predictions).
  - `requirements.txt`: Project dependencies.

- `RAG/`: Evaluation for this use case.

  - `dummy_agent/`: Example implementation of an agent-based system.

    - `agent_langchain.py`: Agent logic, integration with Azure OpenAI and ChromaDB.
    - `main.py`: Application entry point.
    - `create_chromadb_and_load_knowledge_base.py`: Script to create the ChromaDB vector database.
    - `chroma_db/`: Local persistence of the ChromaDB vector database for the implemented example (created when the setup script is executed).

  - `requirements.txt`: Project dependencies.
  - `mlflow_eval.py`: Evaluation script to be executed, with integration of RAGAS metrics.


## Create virtual enviroment and install requirements

1. Upgrade pip
    ```
    python3 -m pip install --upgrade pip
    ```

2. Install virtualenv if you haven't installed it yet.
    ```
    pip install virtualenv
    ```

3. create a virtual enviroment
    ```
    cd [use_case_name]
    python -m venv .venv
    ```

    `use_case_name` can be one of: [ Classification, QnA, RAG ]


4. Create `[use_case_name]/.env` file using `.env_template` as a guide. Set enviroment variable values.

    **Note: read the instructions carefully in .env_template**

5. Start up enviroment
    ```
    source .venv/Scripts/activate   # En Windows: .venv\Scripts\activate
    ```

## Install requirements and Run evaluation cases

**a. Classification**

It is a static evaluation (with predictions made previously).

This case requires a predictions file located in "evaluations/{classification_type}/df_{classification_type}_mlflow_eval_predictions.xlsx", when classification_type can be a necessary business classification, e.g., "first classification step", "region", "is_of_interest", "economic_activity", etc.

```
cd Classification/
pip install -r requirements.txt
python mlflow_eval.py
mlflow ui --port 8080
```

**b. QnA (Question-Answering)**

* b.1. In a terminal tab

    ```
    cd QnA/
    pip install -r requirements.txt
    mlflow server --host 127.0.0.1 --port 8080
    ```

* b.2. In another terminal tab

  - Static evaluation

    ```
    cd QnA/
    python mlflow_static_eval.py
    mlflow ui --port 8080
    ```

  - Dynamic evaluation

    ```
    cd QnA/
    python mlflow_dynamic_eval.py
    mlflow ui --port 8080
    ```  

**c. RAG**

* c.1. In a terminal tab

    ```
    cd RAG/
    pip install -r requirements.txt
    cd dummy_agent/src/
    python create_chromadb_and_load_knowledge_base.py
    cd ../../
    mlflow server --host 127.0.0.1 --port 8080
    ```

* c.2. In another terminal tab

    ```
    cd RAG/
    python mlflow_eval.py
    mlflow ui --port 8080
    ```

## Good Practices

* LLM-as-Judge metrics (Builtin) have different scales depending on the framework (MLFlow 1-5, Ragas 0-1). To compare equivalent results, transform the scales
* Read the repository's Readme before using and consult the necessary documentation
* Before adding a metric to the template, consider: definitions, possible errors, and adaptation to each use case of this template. If library versions are changed or new ones are added, update the requirements file with the specific version
* For development purposes, logging is not used within the mlflow runs module because it doesn't print the logging data; that's why print is used instead.