import os

# ---------------------- Database Configuration ---------------------
# PostgreSQL database configuration using environment variables
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "ml_engineering")
# Utilisateur principal (postgres) ou utilisateur dédié (ml_user)
POSTGRES_USER = os.getenv(
    "POSTGRES_USER", "postgres"
)  # Changez en "ml_user" pour utiliser l'utilisateur dédié
POSTGRES_PASSWORD = os.getenv(
    "POSTGRES_PASSWORD", "password"
)  # Changez en "ml_password" pour l'utilisateur dédié

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# ---------------------- MLflow Configuration ---------------------
# MLflow tracking URI configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", os.path.abspath("./mlruns"))


# ---------------------- Raw Column names ------------------
TRANSACTION_DATE = "date_transaction"
TRANSACTION_MONTH = "mois_transaction"
TRANSACTION_YEAR = "annee_transaction"
DEPARTEMENT = "departement"
REGION = "region"
CITY_UNIQUE_ID = ["departement", "ville", "id_ville"]
SURFACE = "surface_habitable"
PRICE_PER_SQUARE_METER = "prix_m2"

# ---------------------- Feature & Target Column names ------------------
AVERAGE_PRICE_PER_SQUARE_METER = "prix_m2_moyen"
NB_TRANSACTIONS_PER_MONTH = "nb_transactions_mois"
VEFA = "vefa"

REGRESSION_TARGET = "prix"
CLASSIFICATION_TARGET = "en_dessous_du_marche"

random_state = 42

# %%
