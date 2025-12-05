from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests

def call_fastapi_train(model_name):
    """
    Entraîne un modèle spécifique via l'API FastAPI.
    
    Args:
        model_name (str): Nom du modèle à entraîner (rf, svm, logreg)
    """
    url = f"http://server:8000/train?model={model_name}"
    print(f"-----------------Calling: {url}")
    
    try:
        response = requests.get(url)
        print(f"-----------------Response Status: {response.status_code}")
        print(f"-----------------Response Body: {response.text}")
        response.raise_for_status()
        
        # Récupérer les métriques
        data = response.json()
        accuracy = data.get("metrics", {}).get("accuracy", "N/A")
        print(f"-----------------Model {model_name} trained successfully!")
        print(f"-----------------Accuracy: {accuracy}")
        
    except requests.exceptions.RequestException as e:
        print(f"-----------------Error training model {model_name}: {e}")
        raise


# Définir le DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id="train_iris_models_parallel",
    default_args=default_args,
    description='Entraîne les 3 modèles Iris en PARALLÈLE toutes les 2 minutes',
    schedule_interval="*/2 * * * *",  # Toutes les 2 minutes
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['ml', 'iris', 'training', 'parallel'],
) as dag:

    # Tâche 1 : Entraîner RandomForest
    train_rf = PythonOperator(
        task_id="train_random_forest",
        python_callable=call_fastapi_train,
        op_kwargs={'model_name': 'rf'},
    )

    # Tâche 2 : Entraîner SVM
    train_svm = PythonOperator(
        task_id="train_svm",
        python_callable=call_fastapi_train,
        op_kwargs={'model_name': 'svm'},
    )

    # Tâche 3 : Entraîner Logistic Regression
    train_logreg = PythonOperator(
        task_id="train_logistic_regression",
        python_callable=call_fastapi_train,
        op_kwargs={'model_name': 'logreg'},
    )