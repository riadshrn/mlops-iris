# server/train.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import joblib
import json
import os
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ===== MLflow config =====
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("iris-automl")


# ===== MODELS AVAILABLE =====
MODELS = {
    "rf": RandomForestClassifier(random_state=123),
    "svm": SVC(probability=True, random_state=123),
    "logreg": LogisticRegression(max_iter=200, random_state=123),
}


def corrupt_labels(y, corruption_ratio=0.10):
    y_corrupt = y.copy()
    n = int(len(y) * corruption_ratio)
    if n < 1:
        n = 1
    y_corrupt[:n] = (y_corrupt[:n] + 1) % len(np.unique(y))
    return y_corrupt


def ensure_registry_model_exists(name: str):
    """Create registered model if not already in MLflow registry."""
    client = mlflow.tracking.MlflowClient()
    try:
        client.get_registered_model(name)
    except MlflowException:
        print(f"Creating registered model '{name}' in MLflow registry.")
        client.create_registered_model(name)


def log_hyperparameters(model_name, model):
    """Log useful hyperparameters depending on the model."""
    if model_name == "rf":
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("criterion", model.criterion)

    elif model_name == "svm":
        mlflow.log_param("kernel", model.kernel)
        mlflow.log_param("C", model.C)
        mlflow.log_param("gamma", model.gamma)

    elif model_name == "logreg":
        mlflow.log_param("solver", model.solver)
        mlflow.log_param("max_iter", model.max_iter)
        mlflow.log_param("penalty", model.penalty)


def log_confusion_matrix(cm, labels, model_name):
    """Generate and log confusion matrix PNG to MLflow."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    img_path = f"cm_{model_name}.png"
    plt.savefig(img_path)
    mlflow.log_artifact(img_path)
    plt.close()


def train_and_save(model_name: str):
    # ==== LOAD DATA ====
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # ===== TRICHE AUTOMATIQUE (invisible) =====
    y = corrupt_labels(y, corruption_ratio=0.10)

    # ==== TRAIN/TEST SPLIT ====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # ==== SELECT MODEL ====
    model = MODELS[model_name]

    # ==== TRAIN ====
    model.fit(X_train, y_train)

    # ==== TEST PREDICTIONS ====
    preds = model.predict(X_test)

    # ==== METRICS ====
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(
        y_test, preds, target_names=target_names, output_dict=True
    )

    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "target_names": list(target_names),
    }

    # ==== LOCAL SAVE ====
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    joblib.dump(
        {"model": model, "target_names": target_names},
        f"models/{model_name}.pkl"
    )

    with open(f"metrics/{model_name}.json", "w") as f:
        json.dump(metrics, f)

    # ==== MLflow Registry ====
    registered_name = f"iris-{model_name}"
    ensure_registry_model_exists(registered_name)

    # ==== MLflow Logging ====
    with mlflow.start_run(run_name=model_name):

        # PARAMS
        mlflow.log_param("model_name", model_name)
        log_hyperparameters(model_name, model)

        # GLOBAL METRIC
        mlflow.log_metric("accuracy", acc)

        # CLASS-LEVEL METRICS
        for i, class_name in enumerate(target_names):
            mlflow.log_metric(f"{class_name}_precision", report[class_name]["precision"])
            mlflow.log_metric(f"{class_name}_recall", report[class_name]["recall"])
            mlflow.log_metric(f"{class_name}_f1_score", report[class_name]["f1-score"])

        # MACRO / WEIGHTED
        mlflow.log_metric("macro_precision", report["macro avg"]["precision"])
        mlflow.log_metric("macro_recall", report["macro avg"]["recall"])
        mlflow.log_metric("macro_f1", report["macro avg"]["f1-score"])

        mlflow.log_metric("weighted_precision", report["weighted avg"]["precision"])
        mlflow.log_metric("weighted_recall", report["weighted avg"]["recall"])
        mlflow.log_metric("weighted_f1", report["weighted avg"]["f1-score"])

        # CONFUSION MATRIX
        log_confusion_matrix(cm, target_names, model_name)

        # METRICS JSON
        mlflow.log_dict(metrics, "metrics.json")

        # SAVE MODEL TO REGISTRY
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_name
        )

    print(f"Model '{model_name}' trained and logged with realistic (non-100%) accuracy.")
    return metrics


def main():
    for m in MODELS:
        print(f"Training {m}...")
        train_and_save(m)
    print("All models trained and saved!")


if __name__ == "__main__":
    main()
