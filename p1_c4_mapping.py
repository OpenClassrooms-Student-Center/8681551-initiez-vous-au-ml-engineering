# %%
import json
import mlflow
from mlflow import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import Enum
import pandas as pd
from settings import MLFLOW_TRACKING_URI


# %%
print(f"Setting MLflow tracking URI to: {MLFLOW_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# %%
print("Searching for MLflow runs...")
runs = mlflow.search_runs(
    search_all_experiments=True, order_by=["metrics.average_test_f1 DESC"]
)
print(f"Found {len(runs)} runs")

# %%
best_run_id = runs.iloc[0]["run_id"]
print(f"Best run ID: {best_run_id}")

# %%
# Load the model using the runs:/ URI
model_uri = f"runs:/{best_run_id}/catboost_classifier"
print(f"Loading model from: {model_uri}")
model = mlflow.sklearn.load_model(model_uri)
print("Model loaded successfully!")


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
# Initialize FastAPI app with comprehensive documentation
app = FastAPI(
    title="API de Prédiction des Prix Immobiliers",
    description="""
API de Prédiction des Prix Immobiliers

Cette API fournit des prédictions d'apprentissage automatique pour les prix immobiliers en utilisant différents types de modèles.

**Types de Modèles Disponibles :**
- **classification** : Prédit les catégories de prix (ex: bas, moyen, haut)
- **regression** : Prédit les valeurs de prix exactes (bientôt disponible !)

**Fonctionnalités :**
- Paramètre de chemin pour spécifier le type de modèle (classification/regression)
- Paramètres de requête pour des options supplémentaires
- Validation d'entrée complète avec les modèles Pydantic
- Scores de probabilité optionnels pour les modèles de classification
- Documentation interactive de l'API via Swagger UI

**Utilisation :**
1. Choisissez votre type de modèle dans le chemin (ex: `/predict/classification`)
2. Optionnellement, définissez les paramètres de requête (ex: `?include_probability=true`)
3. Envoyez les caractéristiques de la propriété dans le corps de la requête
4. Recevez les prédictions avec les scores de confiance
    """,
    version="1.0.0",
    contact={
        "name": "Équipe d'Ingénierie ML",
        "email": "ml-team@company.com",
    },
    license_info={
        "name": "MIT",
    },
)


# -------------------------- Enums --------------------------


class TypeBatiment(str, Enum):
    """Types de bâtiment disponibles"""

    APPARTEMENT = "Appartement"
    MAISON = "Maison"


class PlanVefa(str, Enum):
    """Options pour le plan VEFA"""

    OUI = "Oui"
    NON = "Non"


# -------------------------- Pydantic Models --------------------------


class PropertyFeaturesBusiness(BaseModel):
    """
    Caractéristiques de propriété pour la prédiction des prix immobiliers (modèle métier).

    Ce modèle définit la structure des données de propriété avec des champs intuitifs pour les utilisateurs métier.
    Toutes les caractéristiques sont optionnelles pour permettre une entrée flexible, mais le modèle utilisera
    des valeurs par défaut pour les caractéristiques manquantes.
    """

    # Caractéristiques du bâtiment
    type_batiment: TypeBatiment = Field(
        default=TypeBatiment.APPARTEMENT,
        alias="Type de Bâtiment",
        description="Type de bâtiment : Appartement ou Maison",
    )

    surface_habitable: float = Field(
        default=50.0,
        alias="Surface Habitable (m²)",
        description="Surface habitable en mètres carrés",
        ge=0,
        le=1000,
    )

    # Indicateurs de marché
    prix_m2_moyen_mois_precedent: float = Field(
        default=3000.0,
        alias="Prix Moyen au m² (€)",
        description="Prix moyen par mètre carré du mois précédent",
        ge=0,
        le=50000,
    )

    nb_transactions_mois_precedent: int = Field(
        default=10,
        alias="Nombre de Transactions",
        description="Nombre de transactions du mois précédent",
        ge=0,
        le=1000,
    )

    # Facteurs économiques
    taux_interet: float = Field(
        default=2.5,
        alias="Taux d'Intérêt (%)",
        description="Taux d'intérêt en pourcentage",
        ge=0,
        le=20,
    )

    variation_taux_interet: float = Field(
        default=0.1,
        alias="Variation Taux d'Intérêt",
        description="Variation du taux d'intérêt",
        ge=-5,
        le=5,
    )

    acceleration_taux_interet: float = Field(
        default=0.05,
        alias="Accélération Taux d'Intérêt",
        description="Accélération du taux d'intérêt",
        ge=-2,
        le=2,
    )

    # Caractéristiques géographiques
    longitude: float = Field(
        default=2.3522,
        alias="Longitude",
        description="Coordonnée de longitude (ex: 2.3522 pour Paris)",
        ge=-180,
        le=180,
    )

    latitude: float = Field(
        default=48.8566,
        alias="Latitude",
        description="Coordonnée de latitude (ex: 48.8566 pour Paris)",
        ge=-90,
        le=90,
    )

    plan_vefa: PlanVefa = Field(
        default=PlanVefa.NON,
        alias="Plan VEFA",
        description="Plan VEFA (Vente en l'État Futur d'Achèvement) : Oui ou Non",
    )

    class Config:
        """Configuration Pydantic pour une meilleure documentation"""

        schema_extra = {
            "example": {
                "Type de Bâtiment": "Appartement",
                "Surface Habitable (m²)": 75.5,
                "Prix Moyen au m² (€)": 3500.0,
                "Nombre de Transactions": 15,
                "Taux d'Intérêt (%)": 3.2,
                "Variation Taux d'Intérêt": 0.15,
                "Accélération Taux d'Intérêt": 0.08,
                "Longitude": 2.3522,
                "Latitude": 48.8566,
                "Plan VEFA": "Non",
            }
        }


class PropertyFeatures(BaseModel):
    """
    Caractéristiques de propriété techniques pour le modèle ML.

    Ce modèle interne est utilisé pour la conversion des données métier vers le format attendu par le modèle.
    """

    # Caractéristiques du bâtiment (encodage one-hot)
    type_batiment_Appartement: int = 0
    """Type de bâtiment : 1 pour Appartement, 0 pour autres types"""

    surface_habitable: float = 50.0
    """Surface habitable en mètres carrés (par défaut : 50.0)"""

    # Indicateurs de marché
    prix_m2_moyen_mois_precedent: float = 3000.0
    """Prix moyen par mètre carré du mois précédent (par défaut : 3000.0)"""

    nb_transactions_mois_precedent: int = 10
    """Nombre de transactions du mois précédent (par défaut : 10)"""

    # Facteurs économiques
    taux_interet: float = 2.5
    """Taux d'intérêt en pourcentage (par défaut : 2.5)"""

    variation_taux_interet: float = 0.1
    """Variation du taux d'intérêt (par défaut : 0.1)"""

    acceleration_taux_interet: float = 0.05
    """Accélération du taux d'intérêt (par défaut : 0.05)"""

    # Caractéristiques géographiques
    longitude: float = 2.3522
    """Coordonnée de longitude (par défaut : longitude de Paris)"""

    latitude: float = 48.8566
    """Coordonnée de latitude (par défaut : latitude de Paris)"""

    vefa: int = 0
    """Indicateur VEFA (Vente en l'État Futur d'Achèvement) : 1 pour VEFA, 0 pour vente classique (par défaut : 0)"""


class ClassificationResponse(BaseModel):
    """
    Modèle de réponse pour les prédictions de classification.

    Retourne la catégorie de prix prédite et les scores de probabilité optionnels.
    """

    prediction: int
    """Catégorie de prix prédite (0: bas, 1: moyen, 2: élevé)"""

    probability: float
    """Score de confiance pour la prédiction (0.0 à 1.0)"""

    model_run_id: str
    """ID de run MLflow du modèle utilisé pour la prédiction"""

    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.85,
                "model_run_id": "abc123def456",
            }
        }


# -------------------------- Mapping Functions --------------------------


def map_business_to_technical_features(
    business_features: PropertyFeaturesBusiness,
) -> PropertyFeatures:
    """
    Convertit les caractéristiques métier en caractéristiques techniques pour le modèle ML.

    Args:
        business_features: Caractéristiques au format métier

    Returns:
        PropertyFeatures: Caractéristiques au format technique pour le modèle
    """
    # Mapping du type de bâtiment
    type_batiment_Appartement = (
        1 if business_features.type_batiment == TypeBatiment.APPARTEMENT else 0
    )

    # Mapping du plan VEFA
    vefa = 1 if business_features.plan_vefa == PlanVefa.OUI else 0

    return PropertyFeatures(
        type_batiment_Appartement=type_batiment_Appartement,
        surface_habitable=business_features.surface_habitable,
        prix_m2_moyen_mois_precedent=business_features.prix_m2_moyen_mois_precedent,
        nb_transactions_mois_precedent=business_features.nb_transactions_mois_precedent,
        taux_interet=business_features.taux_interet,
        variation_taux_interet=business_features.variation_taux_interet,
        acceleration_taux_interet=business_features.acceleration_taux_interet,
        longitude=business_features.longitude,
        latitude=business_features.latitude,
        vefa=vefa,
    )


# -------------------------- API Endpoints --------------------------


@app.get("/", tags=["Général"])
async def root_advanced():
    """
    Point de terminaison racine fournissant les informations de base de l'API.

    Returns:
        dict: Informations de base de l'API et statut
    """
    return {
        "message": "API de Prédiction des Prix Immobiliers",
        "status": "running",
        "version": "1.0.0",
        "available_endpoints": {
            "classification": "/predict/classification",
            "regression": "/predict/regression (bientôt disponible)",
            "health": "/health",
            "docs": "/docs",
        },
    }


@app.get("/health", tags=["Général"])
async def health_check():
    """
    Point de terminaison de vérification de santé pour vérifier le statut de l'API et du modèle.

    Returns:
        dict: Statut de santé incluant le statut de chargement du modèle
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "catboost_classifier",
        "features_count": len(feature_names) if feature_names else 0,
    }


@app.post(
    "/predict/classification",
    response_model=ClassificationResponse,
    tags=["Classification"],
)
async def predict_classification(
    features: PropertyFeaturesBusiness,
    include_probability: bool = True,
    model_run_id: str = None,
):
    """
    Prédire la catégorie de prix immobilier en utilisant un modèle de classification.

    Ce point de terminaison utilise un classificateur CatBoost entraîné pour prédire si une propriété
    tombe dans les catégories de prix bas, moyen ou élevé basées sur ses caractéristiques.

    Les caractéristiques sont saisies dans un format métier intuitif et converties automatiquement
    vers le format technique requis par le modèle.

    Args:
        features (PropertyFeaturesBusiness): Caractéristiques de la propriété au format métier
        include_probability (bool, optional): S'il faut inclure les scores de probabilité. Par défaut True.
        model_run_id (str, optional): ID de run de modèle spécifique à utiliser. Par défaut le meilleur modèle.

    Returns:
        ClassificationResponse: Résultats de prédiction avec catégorie et score de confiance

    Raises:
        HTTPException: Si le modèle n'est pas chargé ou si la prédiction échoue
    """
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Modèle de classification non chargé. Veuillez vérifier l'initialisation du modèle.",
        )

    # Convertir les caractéristiques métier vers le format technique
    technical_features = map_business_to_technical_features(features)

    # Convert features to DataFrame format expected by the model
    features_dict = technical_features.model_dump()
    features_df = pd.DataFrame([features_dict])

    # Ensure all required features are present
    missing_features = set(feature_names) - set(features_df.columns)
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Caractéristiques requises manquantes : {list(missing_features)}. "
            f"Caractéristiques requises : {feature_names}",
        )

    # Select only the features used by the model
    features_for_prediction = features_df[feature_names]

    # Make prediction
    prediction = model.predict(features_for_prediction)[0]

    if include_probability:
        probability = model.predict_proba(features_for_prediction)[0].max()

    return ClassificationResponse(
        prediction=int(prediction),
        probability=float(probability),
        model_run_id=model_run_id or best_run_id,
    )


# %%
