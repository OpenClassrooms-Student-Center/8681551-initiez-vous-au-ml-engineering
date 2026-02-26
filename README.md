# Initiez-vous au ML Engineering

Ce dépôt accompagne les screencasts du cours OpenClassrooms **"Initiez-vous au ML Engineering"**.

**Lien du cours** : https://openclassrooms.com/fr/courses/8681551-initiez-vous-au-ml-engineering

## Correspondance fichiers / chapitres du cours

Chaque script Python correspond a une combinaison **Partie + Chapitre** du cours. La convention de nommage est `p{partie}_c{chapitre}.py` :

| Fichier | Partie & Chapitre | Ce que vous apprenez |
|---------|-------------------|----------------------|
| `p1_c3.py` | Partie 1, Chapitre 3 | Créer une API simple avec FastAPI pour servir un modèle ML |
| `p1_c4_mapping.py` | Partie 1, Chapitre 4 | Séparer la logique métier de l'implémentation technique |
| `p1_c4_orm.py` | Partie 1, Chapitre 4 | Intégrer une base de données avec SQLModel pour calculer des features historiques |

Les fichiers sont pensés pour être découverts dans cet ordre. Chaque étape introduit de nouveaux concepts tout en s'appuyant sur les précédents.

## Fichiers de support

| Fichier | Rôle |
|---------|------|
| `models.py` | Modèles SQLModel (définition de la table `HistoricalTransaction`) |
| `settings.py` | Configuration centralisée (base de données, MLflow, noms de colonnes) |
| `init_db.py` | Script d'initialisation de la base de données depuis un fichier parquet |
| `docker-compose.yml` | Orchestration Docker (API + PostgreSQL) |
| `Dockerfile.p2_c1_2` | Dockerfile de production optimisé avec uv |

## Prérequis

- **Python 3.12+**
- **Docker Desktop** (pour le déploiement conteneurisé)
- **uv** (gestionnaire de paquets Python) - [installation](https://docs.astral.sh/uv/)

## Lancer le projet

### Avec Docker

```bash
docker-compose up --build
```

L'API est ensuite accessible sur http://localhost:8000 et la documentation interactive sur http://localhost:8000/docs.

### En local avec uv

```bash
# Installer les dépendances
uv sync

# Démarrer PostgreSQL
docker-compose up postgres -d

# Initialiser la base de données
python init_db.py

# Lancer le serveur
uv run uvicorn p1_c4_orm:app --reload
```

## Variables d'environnement

Le projet utilise des variables d'environnement pour la configuration. Créez un fichier `.env` à la racine du projet :

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=ml_engineering
POSTGRES_USER=postgres
POSTGRES_PASSWORD=<votre_mot_de_passe>
MLFLOW_TRACKING_URI=./mlruns
```

> **Note** : Ne commitez jamais de fichiers `.env` contenant des mots de passe. Le fichier `.env` est déjà exclu via `.gitignore`.

## Architecture

```
Endpoints FastAPI (p1_c4_orm.py)
         │
Couche Logique Métier (p1_c4_mapping.py)
  - Mapping entrées utilisateur → features techniques
  - Calcul des features historiques depuis la DB
         │
Couche Données / Modèle
  - ORM SQLModel (models.py)
  - Chargement du modèle via MLflow
  - Base de données PostgreSQL
```

## Endpoints API

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/` | Métadonnées de l'API |
| `GET` | `/health` | Vérification de santé |
| `POST` | `/predict/classification` | Prédiction de catégorie de prix |

## Commandes utiles

```bash
# Développement
uv run uvicorn p1_c4_orm:app --reload     # Serveur de développement
uv run pytest --cov --cov-report=term-missing  # Tests avec couverture

# Qualité du code
ruff check --fix     # Linting
ruff format --check  # Formatage

# Base de données
python init_db.py              # Initialiser les données
python init_db.py --drop-existing  # Réinitialiser
```

## Stack technique

- **ML** : CatBoost, scikit-learn, MLflow
- **API** : FastAPI, uvicorn
- **DB** : PostgreSQL, SQLModel
- **Data** : pandas, polars, duckdb
- **Dev** : pytest, ruff, uv

## CI/CD

- **CI** (Pull Requests) : Lint Ruff, formatage, tests pytest avec couverture
- **CD** (branche main) : Build et push vers AWS ECR
