import json
import itertools
import logging
import mlflow
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    confusion_matrix,
    multilabel_confusion_matrix,
    ConfusionMatrixDisplay
)

from constants import (
    FIRST_CATEGORIES,
    SECOND_CATEGORIES
)

class NumpyEncoder(json.JSONEncoder):
    """
    Encoder personalizado para manejar tipos numpy y otros objetos no serializables
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


def get_two_divisors(n):
    if n == 2:
        return 1, 2
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return i, n // i
    return 2, round(n/2) + 1


def plot_multilabel_confusion_matrix(y_true, y_pred, classification_type, labels):
    """
    Plots and save a confusion matrix for multilabel classification.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        classification_type (str): Type of classification (e.g., "first", "second").
        labels (list): List of class labels.

    Returns:
        Figure: Matplotlib figure containing the confusion matrix plots.
    """
    os.makedirs(f"evaluations/{classification_type}", exist_ok=True)
    plt.figure(figsize=(16, 12))
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'DejaVu Sans',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })

    confusion_matrix = multilabel_confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    num_labels = len(labels)
    nrows, ncols = get_two_divisors(num_labels)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24, 10))
    fig.suptitle('Multi-Label Confusion Matrix (VP/FP-FN/VN)', fontsize=16, fontweight='bold')

    # Desempaqueta todos los subplots en un arreglo plano
    axe = axes.ravel()

    for i, (cfs_matrix, label) in enumerate(zip(confusion_matrix, labels)):
        disp = ConfusionMatrixDisplay(cfs_matrix, display_labels=[0, 1])
        disp.plot(include_values=True, cmap="Blues", ax=axe[i], xticks_rotation="vertical")
        axe[i].set_title(f"{label}")
        axe[i].set_xlabel("Correcta")
        axe[i].set_ylabel("Predicha")

    plt.tight_layout()
    plt.savefig(f"evaluations/{classification_type}/{classification_type}_confusion_matrix.png", 
                   dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none'
    )


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Binary Confusion matrix (VP/FP-FN/VN)',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[j, i], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel("Correcta")
    plt.ylabel("Predicha")
    plt.savefig(f"evaluations/first/first_confusion_matrix.png", 
                   dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none'
    )


# Replace with classification step names
classification_types = [ "first", "second" ]

for classification_type in classification_types:

    logging.info(f"\n\nCLASIFICACION: {classification_type}\n\n")

    # Replace with each category list names
    if classification_type == "first":
        labels = FIRST_CATEGORIES
    elif classification_type == "second":
        labels = SECOND_CATEGORIES

    mlflow.set_experiment(f"experiment_eval_{classification_type}_classificator")

    with mlflow.start_run(run_name=f"{classification_type}_classifier") as run:
        mlflow.log_param("model_name", "OpenAI_Classifier")
        mlflow.log_param("temperature", 0.0)
        mlflow.log_param("top_p", 0.7)
        mlflow.log_param("max_tokens", 1000)

        os.makedirs("evaluations", exist_ok=True)
 
        try:
            df_eval = pd.read_excel(f"evaluations/{classification_type}/df_{classification_type}_mlflow_eval_predictions.xlsx",
                                        engine="openpyxl")
        except FileNotFoundError as e:
            logging.error(f"File df_{classification_type}_mlflow_eval_predictions.xlsx not found in evaluations/{classification_type}/")

        if classification_type == "first":
            # asume that "first" classification is binary
            evaluator_config = {
                "labels": labels,
                "pos_label": 'Sí',
            }
        else:
            evaluator_config = {
                "label_list": labels
            }

        result = mlflow.evaluate(
            data=df_eval,
            predictions="prediction",
            targets="target",
            model_type="classifier",
            evaluator_config=evaluator_config,
        )

        y_true = df_eval["target"].tolist()
        y_pred = df_eval["prediction"].tolist()

        report = classification_report(y_true=y_true, y_pred=y_pred, labels=labels, output_dict=True)
        df_report = pd.DataFrame(report).transpose()

        # Check if any class has not samples predicted
        for class_label, metrics in report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                if metrics['precision'] == 0.0 and metrics['support'] > 0:
                    logging.warning(f"Class '{class_label}' has ill-defined precision (0.0) due to no predicted samples.")

        mlflow.log_metrics(
            {
                "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
                "Latency hours": (df_eval["total_latency_minutes"].mean())/60,
            }
        )

        if classification_type == "first":
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            plot_confusion_matrix(cm, labels)
        else:
            plot_multilabel_confusion_matrix(
                y_true, y_pred,
                classification_type,
                labels
            )

        df_report.to_excel(f"evaluations/{classification_type}/{classification_type}_classification_report.xlsx", 
                           index=True, engine="openpyxl")

        with open(f"evaluations/{classification_type}/df_{classification_type}_eval_results.json", 
                  "w", encoding="utf-8") as f:
            json.dump(result.metrics, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        mlflow.log_artifact(f"evaluations/{classification_type}/df_{classification_type}_eval_results.json")
        mlflow.log_artifact(f"evaluations/{classification_type}/{classification_type}_classification_report.xlsx")
        mlflow.log_artifact(f"evaluations/{classification_type}/{classification_type}_confusion_matrix.png")
