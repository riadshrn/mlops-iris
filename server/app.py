# server/app.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
import numpy as np
import joblib
import json
import os

from train import train_and_save 
import mlflow

mlflow.set_tracking_uri("http://mlflow:5000")

app = FastAPI(title="Iris AutoML API")

AVAILABLE_MODELS = ["rf", "svm", "logreg"]

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    model: str
    predicted_class_index: int
    predicted_class_name: str
    probabilities: list[float]
    class_labels: list[str]


def ensure_model_exists(model_name: str):
    model_path = f"models/{model_name}.pkl"
    if not os.path.exists(model_path):
        print(f"Model {model_name} not found, training...")
        train_and_save(model_name)

def load_model(model_name: str):
    ensure_model_exists(model_name)
    return joblib.load(f"models/{model_name}.pkl")

def load_metrics(model_name: str):
    metrics_path = f"metrics/{model_name}.json"
    if not os.path.exists(metrics_path):
        train_and_save(model_name)
    with open(metrics_path, "r") as f:
        return json.load(f)

@app.get("/models")
def list_models():
    return {"available_models": AVAILABLE_MODELS}

@app.get("/train")
def train(model: str = Query("rf", enum=AVAILABLE_MODELS)):
    metrics = train_and_save(model)
    return {"message": f"Model '{model}' trained", "metrics": metrics}

@app.get("/metrics")
def metrics(model: str = Query("rf", enum=AVAILABLE_MODELS)):
    metrics = load_metrics(model)
    return {"model": model, "metrics": metrics}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures, model: str = Query("rf", enum=AVAILABLE_MODELS)):
    data = jsonable_encoder(features)
    x = np.array([[
        data["sepal_length"],
        data["sepal_width"],
        data["petal_length"],
        data["petal_width"],
    ]])

    payload = load_model(model)
    clf = payload["model"]
    target_names = payload["target_names"]

    y_pred = clf.predict(x)[0]
    y_proba = clf.predict_proba(x)[0].tolist()

    return PredictionResponse(
        model=model,
        predicted_class_index=int(y_pred),
        predicted_class_name=str(target_names[y_pred]),
        probabilities=y_proba,
        class_labels=target_names.tolist()
    )

@app.get("/update-model")
def update_model(model: str, version: int):
    global current_model

    registered_name = f"iris-{model}"
    model_uri = f"models:/{registered_name}/{version}"
    current_model = mlflow.pyfunc.load_model(model_uri)

    return {"message": f"Model {model} v{version} loaded"}
