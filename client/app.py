import os
import streamlit as st
import requests
import pandas as pd

API_BASE_URL = "http://server:8000"  # FastAPI
MLFLOW_URL = "http://mlflow:5000"    # MLflow REST API

IMAGE_PATHS = {
    "setosa": "images/setosa.jpg",
    "versicolor": "images/versicolor.jpg",
    "virginica": "images/virginica.jpg",
}

st.set_page_config(page_title="Iris AutoML", layout="wide")
st.title("Iris AutoML – MLOps Demo")
st.markdown("Entraînez plusieurs modèles sur Iris, comparez leurs métriques, mettez à jour via MLflow et faites des prédictions visuelles.")


# ===============================================================
# FONCTION : récupérer tous les modèles & versions MLflow
# ===============================================================
def fetch_mlflow_models():
    """Retourne un dict {model_name: [versions]} depuis MLflow Registry."""
    try:
        # CORRECTION: Utiliser l'API de Registered Models au lieu de Logged Models
        url = f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/search"
        print("📡 Requête envoyée à :", url)

        resp = requests.get(url, params={"max_results": 100})
        print("📡 Statut HTTP :", resp.status_code)
        print("📡 Réponse brute :", resp.text)

        if resp.status_code != 200:
            print("❌ Erreur HTTP :", resp.status_code)
            return {}

        data = resp.json()
        print("📡 JSON décodé :", data)

        models_info = {}

        # API Registered Models → "registered_models"
        for model in data.get("registered_models", []):
            model_name = model["name"]
            
            # Récupérer les versions de ce modèle
            versions_url = f"{MLFLOW_URL}/api/2.0/mlflow/model-versions/search"
            versions_resp = requests.get(
                versions_url,
                params={"filter": f"name='{model_name}'"}
            )
            
            if versions_resp.status_code == 200:
                versions_data = versions_resp.json()
                versions = [
                    int(v["version"]) 
                    for v in versions_data.get("model_versions", [])
                ]
                models_info[model_name] = sorted(versions)
            else:
                print(f"⚠️ Impossible de récupérer les versions pour {model_name}")
                models_info[model_name] = []

        print("📦 Modèles trouvés :", models_info)
        return models_info

    except Exception as e:
        print("❌ Erreur récupération modèles MLflow :", e)
        import traceback
        traceback.print_exc()
        st.sidebar.error(f"Erreur récupération modèles MLflow : {e}")
        return {}




# ===============================================================
# SIDEBAR : CHOIX DU MODÈLE + TRAINING
# ===============================================================
st.sidebar.header("Modèle & Entraînement")

model_label_to_name = {
    "RandomForest (rf)": "rf",
    "SVM (svm)": "svm",
    "Logistic Regression (logreg)": "logreg",
}

model_label = st.sidebar.selectbox("Choisissez un modèle", list(model_label_to_name.keys()))
selected_model = model_label_to_name[model_label]


# ---- Réentraînement local ----
if st.sidebar.button("Réentraîner ce modèle"):
    try:
        r = requests.get(f"{API_BASE_URL}/train", params={"model": selected_model})
        if r.status_code == 200:
            data = r.json()
            st.sidebar.success(
                f"Modèle '{selected_model}' réentraîné (accuracy = {data['metrics']['accuracy']:.3f})"
            )
        else:
            st.sidebar.error(f"Erreur entraînement : {r.text}")
    except Exception as e:
        st.sidebar.error(f"Erreur de connexion API : {e}")


# ---- Afficher les métriques ----
if st.sidebar.button("Afficher les métriques"):
    try:
        r = requests.get(f"{API_BASE_URL}/metrics", params={"model": selected_model})
        if r.status_code == 200:
            metrics = r.json()["metrics"]
            st.session_state["last_metrics"] = metrics
            st.sidebar.info(f"Accuracy {selected_model} : {metrics['accuracy']:.3f}")
        else:
            st.sidebar.error(f"Erreur : {r.text}")
    except Exception as e:
        st.sidebar.error(f"Erreur de connexion API : {e}")


# ===============================================================
# PARTIE C : MISE À JOUR DU MODÈLE DEPUIS MLFLOW
# ===============================================================
st.sidebar.markdown("---")
st.sidebar.header("Mise à jour depuis MLflow Registry")

mlflow_models = fetch_mlflow_models()

if mlflow_models:
    selected_mlflow_model = st.sidebar.selectbox(
        "Modèle MLflow",
        list(mlflow_models.keys())
    )

    version = st.sidebar.selectbox(
        "Version du modèle",
        mlflow_models[selected_mlflow_model]
    )

    if st.sidebar.button("Charger modèle MLflow"):
        try:
            # Ex: iris-rf → rf
            short_name = selected_mlflow_model.replace("iris-", "")

            r = requests.get(
                f"{API_BASE_URL}/update-model",
                params={"model": short_name, "version": int(version)}
            )

            if r.status_code == 200:
                st.sidebar.success(
                    f"Modèle MLflow '{selected_mlflow_model}' v{version} chargé avec succès !"
                )
                
                try:
                    metrics_r = requests.get(
                        f"{API_BASE_URL}/metrics", 
                        params={"model": short_name}
                    )
                    if metrics_r.status_code == 200:
                        metrics = metrics_r.json()["metrics"]
                        st.session_state["last_metrics"] = metrics
                        st.sidebar.info(f"Accuracy : {metrics['accuracy']:.3f}")
                    else:
                        st.sidebar.warning("Modèle chargé mais impossible de récupérer les métriques")
                except Exception as e:
                    st.sidebar.warning(f"Modèle chargé mais erreur métriques : {e}")
            else:
                st.sidebar.error(f"Erreur : {r.text}")

        except Exception as e:
            st.sidebar.error(f"Erreur charge MLflow : {e}")

else:
    st.sidebar.warning("Aucun modèle trouvé dans MLflow Registry.")


# ===============================================================
# CARACTÉRISTIQUES DE LA FLEUR
# ===============================================================
st.sidebar.markdown("---")
st.sidebar.header("Caractéristiques de la fleur")

sepal_length = st.sidebar.number_input("Sepal length (cm)", 0.0, 10.0, 5.1)
sepal_width = st.sidebar.number_input("Sepal width (cm)", 0.0, 10.0, 3.5)
petal_length = st.sidebar.number_input("Petal length (cm)", 0.0, 10.0, 1.4)
petal_width = st.sidebar.number_input("Petal width (cm)", 0.0, 10.0, 0.2)


# ===============================================================
# LAYOUT PRINCIPAL : METRICS + PREDICTION
# ===============================================================
col_left, col_right = st.columns([2, 3])


# --------- METRICS ----------
with col_left:
    st.subheader("Métriques du modèle sélectionné")

    metrics = st.session_state.get("last_metrics")
    if metrics:
        st.write(f"**Accuracy** : `{metrics['accuracy']:.3f}`")

        cm = pd.DataFrame(
            metrics["confusion_matrix"],
            columns=metrics["target_names"],
            index=metrics["target_names"],
        )
        st.write("Matrice de confusion :")
        st.dataframe(cm)

        report = pd.DataFrame(metrics["classification_report"]).T
        st.write("Rapport de classification :")
        st.dataframe(report)

    else:
        st.info("Clique sur **'Afficher les métriques'** dans la sidebar.")


# --------- PREDICTION ----------
with col_right:
    st.subheader("Prédiction sur une nouvelle fleur")

    if st.button("Prédire la classe avec le modèle sélectionné"):
        payload = {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width,
        }

        try:
            r = requests.post(
                f"{API_BASE_URL}/predict",
                params={"model": selected_model},
                json=payload
            )

            if r.status_code == 200:
                data = r.json()
                predicted = data["predicted_class_name"]
                proba = data["probabilities"]
                labels = data["class_labels"]

                col_img, col_proba = st.columns([1, 2])

                with col_img:
                    st.markdown(f"### Classe prédite : **{predicted.upper()}**")

                    img_path = IMAGE_PATHS.get(predicted.lower())
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, width=250)

                    idx = labels.index(predicted)
                    st.metric("Confiance", f"{proba[idx] * 100:.2f} %")

                with col_proba:
                    df = pd.DataFrame({
                        "Classe": labels,
                        "Probabilité (%)": [p * 100 for p in proba],
                    })
                    st.dataframe(df, hide_index=True)
                    st.bar_chart(df.set_index("Classe"))

            else:
                st.error(f"Erreur API : {r.text}")

        except Exception as e:
            st.error(f"Erreur connexion API : {e}")

    else:
        st.info("Règle les valeurs puis clique sur **Prédire**.")