# %%

import polars as pl
import json
import os
import mlflow
from mlflow import MlflowClient
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from settings import PROJECT_PATH

# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# %%

transactions = pl.read_parquet(
    os.path.join(PROJECT_PATH, "transactions_post_feature_engineering.parquet")
)


# %%

# %%
with open("features_used.json", "r") as f:
    feature_names = json.load(f)

with open("categorical_features_used.json", "r") as f:
    categorical_features = json.load(f)

# %%

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


# %%

# Parse the JSON strings to get the actual feature lists
feature_names = eval(features_used)
categorical_features = eval(categorical_features_used)

# %%
feature_names

# %%
# -------------------------- SIMPLE FastAPI Application (Beginner Version) --------------------------

app_simple = FastAPI(
    title="API Simple de Prédiction de Prix Immobilier",
    description="Une API conviviale pour la prédiction de prix immobiliers utilisant la classification",
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
# -------------------------- ADVANCED FastAPI Application (Production Version) --------------------------

'''


@app.post("/predict/regression", response_model=RegressionResponse, tags=["Regression"])
async def predict_regression(features: PropertyFeatures, model_run_id: str = None):
    """
    Predict real estate price value using regression model.

    **Note**: This endpoint is currently under development.
    The regression model will be available in a future version.

    Args:
        features (PropertyFeatures): Property characteristics and market indicators
        model_run_id (str, optional): Specific model run ID to use. Defaults to best model.

    Returns:
        RegressionResponse: Prediction results with price value

    Raises:
        HTTPException: If regression model is not available
    """
    raise HTTPException(
        status_code=501,
        detail="Regression prediction is not yet implemented. "
        "Please use the classification endpoint for now. "
        "Regression models will be available in version 2.0.",
    )

'''
# %%
# -------------------------- Run FastAPI Server --------------------------
"""
if __name__ == "__main__":
    print("Choose which API to run:")
    print("1. Simple API (beginner-friendly): python p1_c3.py simple")
    print("2. Advanced API (production-ready): python p1_c3.py advanced")
    print()

    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "simple":
        print("Starting SIMPLE FastAPI server...")
        print("API will be available at: http://localhost:8000")
        print("API documentation at: http://localhost:8000/docs")
        uvicorn.run(app_simple, host="0.0.0.0", port=8000)
    else:
        print("Starting ADVANCED FastAPI server...")
    print("API will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

# %%
# -------------------------- Usage Examples --------------------------

"""
Example usage of both API versions:

## SIMPLE API (Beginner-Friendly)
Perfect for learning FastAPI basics - no async, no complex features

1. Start the simple server: python p1_c3.py simple

2. Test with curl:
curl -X POST "http://localhost:8000/predict?include_probability=true" \
     -H "Content-Type: application/json" \
     -d '{
       "type_batiment_Appartement": 1,
       "surface_habitable": 75.5,
       "prix_m2_moyen_mois_precedent": 3500.0,
       "nb_transactions_mois_precedent": 15,
       "taux_interet": 3.2,
       "variation_taux_interet": 0.15,
       "acceleration_taux_interet": 0.08,
       "longitude": 2.3522,
       "latitude": 48.8566,
       "vefa": 0
     }'

3. Test without probability:
curl -X POST "http://localhost:8000/predict?include_probability=false" \
     -H "Content-Type: application/json" \
     -d '{
       "surface_habitable": 60.0,
       "prix_m2_moyen_mois_precedent": 2800.0,
       "longitude": 2.3522,
       "latitude": 48.8566,
       "vefa": 0
     }'

4. Interactive docs: http://localhost:8000/docs

## ADVANCED API (Production-Ready)
Full-featured API with async, comprehensive documentation, and multiple endpoints

1. Start the advanced server: python p1_c3.py advanced

2. Test classification endpoint:
curl -X POST "http://localhost:8000/predict/classification?include_probability=true" \
     -H "Content-Type: application/json" \
     -d '{
       "type_batiment_Appartement": 1,
       "surface_habitable": 75.5,
       "prix_m2_moyen_mois_precedent": 3500.0,
       "nb_transactions_mois_precedent": 15,
       "taux_interet": 3.2,
       "variation_taux_interet": 0.15,
       "acceleration_taux_interet": 0.08,
       "longitude": 2.3522,
       "latitude": 48.8566,
       "vefa": 0
     }'

3. Health check: curl http://localhost:8000/health
4. Interactive docs: http://localhost:8000/docs

## Key Differences:
SIMPLE API:
- Uses regular functions (no async/await)
- Uses pandas instead of Polars
- Single /predict endpoint
- Simple dictionary responses
- No health endpoint
- No regression endpoint

ADVANCED API:
- Uses async functions
- Uses Polars for data processing
- Multiple endpoints (/predict/classification, /predict/regression)
- Structured Pydantic response models
- Health check endpoint
- Comprehensive documentation
- Future regression support
"""

# %%
