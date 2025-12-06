# Iris AutoML - MLOps Pipeline Complete

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.0+-0194E2.svg)](https://mlflow.org/)
[![Airflow](https://img.shields.io/badge/Airflow-2.10+-017CEE.svg)](https://airflow.apache.org/)
[![Docker](https://img.shields.io/badge/Docker-20.10+-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Plateforme MLOps complète** pour l'entraînement, le versioning et le déploiement de modèles de classification sur le dataset Iris, avec orchestration automatisée et interface utilisateur interactive.




---

##  Aperçu

**Iris AutoML** est une plateforme MLOps end-to-end qui démontre les meilleures pratiques de Machine Learning Operations à travers un cas d'usage classique : la classification des fleurs Iris.


---

## Architecture

### Vue d'Ensemble

<p align="center">
  <img src="images/ARCHI1.png" alt="Architecture du projet" width="70%">
</p>


## Infrastructure MLOps

<details>
<summary><strong>DATA INGESTION</strong></summary>

Scikit-learn Iris Dataset

</details>

<details>
<summary><strong>MODEL TRAINING (3 modèles)</strong></summary>

- RandomForest Classifier
- Support Vector Machine (SVM)
- Logistic Regression

</details>

<details>
<summary><strong>EXPERIMENT TRACKING</strong></summary>

MLflow (params, metrics, artifacts)

</details>

<details>
<summary><strong>MODEL REGISTRY</strong></summary>

MLflow Model Registry (versioning, staging, production)

</details>

<details>
<summary><strong>MODEL SERVING</strong></summary>

FastAPI REST API

</details>

<details>
<summary><strong>MONITORING & VISUALIZATION</strong></summary>

- MLflow UI (experiments & models)
- Streamlit (user interface)

</details>

<details>
<summary><strong>ORCHESTRATION</strong></summary>

Apache Airflow (scheduled / automated training)

</details>

---


## Pipeline MLOps

<details>
<summary><strong>1. DATA INGESTION</strong></summary>

Iris Dataset (scikit-learn)

</details>

<details>
<summary><strong>2. FEATURE ENGINEERING</strong></summary>

Pas de preprocessing (données déjà normalisées)

</details>

<details>
<summary><strong>3. MODEL TRAINING</strong></summary>

- RandomForest (100 trees, max_depth=None)  
- SVM (RBF kernel, C=1.0)  
- LogisticRegression (max_iter=200)

</details>


<details>
<summary><strong>4. MODEL EVALUATION</strong></summary>

- Accuracy  
- Precision, Recall, F1-Score (par classe)  
- Confusion Matrix  
- Classification Report

</details>


<details>
<summary><strong>5. EXPERIMENT TRACKING (MLflow)</strong></summary>

- Logging des hyperparamètres  
- Logging des métriques  
- Sauvegarde des artefacts (modèle, confusion matrix)  
- Metadata (git commit, user, timestamp)

</details>


<details>
<summary><strong>6. MODEL REGISTRY (MLflow)</strong></summary>

- Enregistrement dans le registry  
- Versioning automatique (v1, v2, v3...)  
- Staging (None, Staging, Production)  
- Transition entre stages

</details>

<details>
<summary><strong>7. MODEL SERVING</strong></summary>

- Chargement via FastAPI  
- Endpoints REST pour prédictions

</details>

<details>
<summary><strong>8. MONITORING</strong></summary>

- MLflow UI (performance tracking)  
- Logs applicatifs (Docker logs)  
- Airflow DAG monitoring

</details>

<details>
<summary><strong>9. RETRAINING (Airflow)</strong></summary>

Scheduled retraining (cron-based)

</details>

---

## Installation

### Installation Rapide

```bash
# 1. Cloner le repository
git clone https://github.com/riadshrn/mlops-iris.git
cd mlops-iris

# 2. Lancer tous les services
docker-compose up --build

# 3. Attendre que tous les services démarrent (~2-3 minutes)
```

### Accès aux Services

Une fois les services démarrés, accédez aux interfaces :

| Service | URL | Credentials |
|---------|-----|-------------|
| **Streamlit UI** | http://localhost:8501 | - |
| **FastAPI Docs** | http://localhost:8000/docs | - |
| **MLflow UI** | http://localhost:5000 | - |
| **Airflow UI** | http://localhost:8080 | admin / admin |

---

## Utilisation

### 01. Via Stramlit

#### Entraîner un Modèle

1. Ouvrir http://localhost:8501
2. Dans la sidebar, sélectionner un modèle (RF, SVM, LogReg)
3. Cliquer sur **"Réentraîner ce modèle"**
4. Attendre le message de succès avec l'accuracy

#### Visualiser les Métriques

1. Sélectionner un modèle
2. Cliquer sur **"Afficher les métriques"**
3. Consulter :
   - Accuracy globale
   - Matrice de confusion
   - Rapport de classification (precision, recall, f1-score)

#### Faire une Prédiction

1. Dans la sidebar "Caractéristiques de la fleur" :
   - Sepal length (cm) : ex. 5.1
   - Sepal width (cm) : ex. 3.5
   - Petal length (cm) : ex. 1.4
   - Petal width (cm) : ex. 0.2
2. Cliquer sur **"Prédire la classe"**
3. Visualiser :
   - Classe prédite avec image
   - Niveau de confiance
   - Distribution des probabilités

#### Charger un Modèle depuis MLflow

1. Section "Mise à jour depuis MLflow Registry"
2. Sélectionner un modèle et une version
3. Cliquer sur **"Charger modèle MLflow"**
4. Les métriques s'affichent automatiquement

### 02. Via MLflow UI

1. Ouvrir http://localhost:5000
2. Naviguer vers **"Experiments"**
3. Sélectionner l'expérience **"iris-automl"**
4. Comparer les runs
5. Consulter les métriques et artefacts

### 03. Via Airflow

1. Ouvrir http://localhost:8080
2. Login : `admin` / `admin`
3. Activer le DAG **"train_iris_model"**
4. Le modèle RF sera réentraîné toutes les minutes
5. Consulter les logs d'exécution

---


### Documentation Interactive

Accédez à la documentation Swagger interactive :
**http://localhost:8000/docs**

---



### Interface MLflow

<p align="center">
  <img src="images/MLFlow-Registred_model.png" alt="Architecture du projet" width="40%">
</p>

### Interface AirFlow

<p align="center">
  <img src="images/Task-duration-airflow.png" alt="Architecture du projet" width="70%">
</p>

---

<div align="center">

**⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile ! ⭐**

Made with ❤️ by [Riad](https://github.com/riadshrn)

</div>