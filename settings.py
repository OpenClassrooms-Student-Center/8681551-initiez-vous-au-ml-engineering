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

# ---------------------- Filepaths ---------------------
# Utilisation de variables d'environnement pour éviter de hardcoder le chemin de la donnée
PROJECT_PATH = os.environ.get("SUPERVISED_LEARNING_PROJECT_PATH")
TRANSACTIONS_FILE_PATH = os.path.join(PROJECT_PATH, "transactions.npz")
FOYER_FISCAUX_FILE_PATH = os.path.join(PROJECT_PATH, "foyers_fiscaux.csv")
TAUX_ENDETTEMENT_FILE_PATH = os.path.join(PROJECT_PATH, "taux_endettement.csv")
ACTIFS_FINANCIERS_FILE_PATH = os.path.join(PROJECT_PATH, "actifs_financiers.csv")
TAUX_DINTERET_FILE_PATH = os.path.join(PROJECT_PATH, "taux_interet.csv")
FLUX_EMPRUNTS_FILE_PATH = os.path.join(PROJECT_PATH, "flux_nouveaux_emprunts.csv")
REGIONS_FILE_PATH = os.path.join(PROJECT_PATH, "departements-france.csv")
INDICE_REFERENCE_LOYERS = os.path.join(PROJECT_PATH, "indice_reference_loyers.csv")


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
