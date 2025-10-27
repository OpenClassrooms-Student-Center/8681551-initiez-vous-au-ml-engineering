"""
Database Initialization Script

This script initializes the database by loading historical transaction data.
It should be run once to populate the database.

Usage:
    python init_db.py

Or in development:
    python init_db.py --drop-existing
"""

import sys
import argparse
import polars as pl
from sqlmodel import SQLModel, create_engine, Session, select
from models import HistoricalTransaction
from settings import (
    DATABASE_URL,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
)

# Database engine configuration
engine = create_engine(DATABASE_URL, echo=False)


def load_historical_data_to_db(drop_existing=False):
    """Charge les données historiques de Nouvelle-Aquitaine depuis le fichier parquet vers la base de données PostgreSQL avec SQLModel"""

    try:
        # Créer les tables
        SQLModel.metadata.create_all(engine)

        # Si drop_existing est True, supprimer les données existantes
        if drop_existing:
            with Session(engine) as session:
                # Supprimer toutes les transactions historiques
                session.exec("TRUNCATE TABLE historicaltransaction;").commit()
                print("Données existantes supprimées.")

        # Vérifier si des données existent déjà
        with Session(engine) as session:
            existing_count = len(session.exec(select(HistoricalTransaction)).all())
            if existing_count > 0:
                print(
                    f"✓ La base de données contient déjà {existing_count} transactions historiques."
                )
                print(
                    "Pour recharger les données, utilisez : python init_db.py --drop-existing"
                )
                return True

        # Charger les données depuis le fichier parquet avec feature engineering
        transactions_df = pl.read_parquet(
            "transactions_post_feature_engineering.parquet"
        )

        # Filtrer uniquement les transactions de Nouvelle-Aquitaine
        transactions_df = transactions_df.filter(
            pl.col("nom_region_Nouvelle-Aquitaine") == 1
        )

        # Convertir en pandas pour SQLModel
        transactions_pd = transactions_df.to_pandas()

        # Mapper les noms de colonnes pour correspondre au modèle SQLModel
        column_mapping = {
            "nom_region_Auvergne-Rhône-Alpes": "nom_region_Auvergne_Rhone_Alpes",
            "nom_region_Provence-Alpes-Côte d'Azur": "nom_region_Provence_Alpes_Cote_d_Azur",
            "nom_region_Île-de-France": "nom_region_Ile_de_France",
            "emprunts_M€": "emprunts_ME",
        }

        transactions_pd = transactions_pd.rename(columns=column_mapping)

        # Insérer les données dans la base avec SQLModel
        with Session(engine) as session:
            # Insérer les transactions historiques
            for _, row in transactions_pd.iterrows():
                transaction = HistoricalTransaction(**row.to_dict())
                session.add(transaction)

            session.commit()
            print(
                f"✓ Chargé {len(transactions_pd)} transactions historiques de Nouvelle-Aquitaine avec feature engineering"
            )

    except FileNotFoundError:
        print(
            "❌ Erreur: Fichier parquet non trouvé. Les données historiques ne sont pas disponibles."
        )
        print(
            "   Assurez-vous que 'transactions_post_feature_engineering.parquet' existe dans le répertoire courant."
        )
        return False
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données: {e}")
        return False

    return True


def main():
    """Point d'entrée principal pour l'initialisation de la base de données"""
    parser = argparse.ArgumentParser(
        description="Initialiser la base de données avec les données historiques"
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Supprimer les données existantes avant de charger les nouvelles données",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Initialisation de la base de données PostgreSQL")
    print("=" * 60)
    print(f"Host: {POSTGRES_HOST}")
    print(f"Port: {POSTGRES_PORT}")
    print(f"Database: {POSTGRES_DB}")
    print(f"User: {POSTGRES_USER}")
    print("-" * 60)

    success = load_historical_data_to_db(drop_existing=args.drop_existing)

    if success:
        print("-" * 60)
        print("✓ Initialisation de la base de données terminée avec succès!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("-" * 60)
        print("❌ L'initialisation de la base de données a échoué.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
