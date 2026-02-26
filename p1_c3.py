# %%
import json
import os
import mlflow
from mlflow import MlflowClient
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd


# %%
# -------------------------- MLflow Model Loading --------------------------

# Set MLflow tracking URI (adjust if needed)
mlruns_path = os.path.abspath("./mlruns")
os.makedirs(mlruns_path, exist_ok=True)
os.makedirs(os.path.join(mlruns_path, ".trash"), exist_ok=True)

mlflow.set_tracking_uri(mlruns_path)
client = MlflowClient(tracking_uri=mlruns_path)

# Search for the best model (you can modify this to select a specific model)
runs = mlflow.search_runs(
    search_all_experiments=True, order_by=["metrics.average_test_f1 DESC"]
)

# %%
runs

# %%
best_run_id = runs.iloc[0]["run_id"]

# %%
runs

# %%
# Load the CatBoost model that was saved using sklearn flavor
# The model was saved with mlflow.sklearn.log_model(), so we need to use sklearn.load_model()
model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/catboost_classifier")

# Load the features used in the best run
run = client.get_run(best_run_id)
features_used = run.data.params.get("features", "[]")
categorical_features_used = run.data.params.get("categorical_features", "[]")

# %%

# Parse the JSON strings to get the actual feature lists
feature_names = json.loads(features_used)
categorical_features = json.loads(categorical_features_used)

# %%
feature_names

# %%
# -------------------------- SIMPLE FastAPI Application (Beginner Version) --------------------------

app_simple = FastAPI(
    title="API Simple de Prédiction de Prix Immobilier",
    description="Une API basique pour la prédiction de prix immobiliers utilisant la classification",
    version="1.0.0",
)


# Simple Pydantic model for input
class SimplePropertyFeatures(BaseModel):
    """Caractéristiques simples d'une propriété pour la prédiction"""

    type_batiment_Appartement: int = 0
    surface_habitable: float = 50.0
    prix_m2_moyen_mois_precedent: float = 3000.0
    nb_transactions_mois_precedent: int = 10
    taux_interet: float = 2.5
    variation_taux_interet: float = 0.1
    acceleration_taux_interet: float = 0.05
    longitude: float = 2.3522  # Par défaut longitude de Paris
    latitude: float = 48.8566  # Par défaut latitude de Paris
    vefa: int = 0  # Par défaut 0 (pas VEFA)


@app_simple.get("/")
def root():
    """Informations de base de l'API"""
    return {
        "message": "API Simple de Prédiction de Prix Immobilier",
        "status": "running",
    }


@app_simple.post("/predict")
def predict_simple(features: SimplePropertyFeatures, include_probability: bool = True):
    """
    Point de terminaison de prédiction simple

    Args:
        features: Caractéristiques de la propriété
        include_probability: S'il faut inclure la probabilité (par défaut: True)

    Returns:
        dict: Résultats de la prédiction
    """

    features_dict = features.model_dump()
    features_df = pd.DataFrame([features_dict])

    # Select only the features used by the model
    features_for_prediction = features_df[feature_names]

    # Make prediction
    prediction = model.predict(features_for_prediction)[0]

    result = {"prediction": int(prediction), "model_run_id": best_run_id}

    if include_probability:
        probability = model.predict_proba(features_for_prediction)[0].max()
        result["probability"] = float(probability)

    return result


# %%
