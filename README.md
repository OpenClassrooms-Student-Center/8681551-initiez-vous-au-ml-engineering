# Initiez-vous au ML Engineering

Ce dépôt accompagne les screencasts du cours OpenClassrooms **"Initiez-vous au ML Engineering"**.

**Lien du cours** : https://openclassrooms.com/fr/courses/8681551-initiez-vous-au-ml-engineering

## Correspondance fichiers / chapitres du cours

Les fichiers du dépôt correspondent aux **Parties et Chapitres** du cours. Les scripts Python suivent la convention `p{partie}_c{chapitre}.py`. Les autres fichiers (Dockerfiles, workflows CI/CD) sont rattachés à la partie correspondante dans le tableau ci-dessous.

### Partie 1 — Servir un modèle ML via une API

| Fichier | Chapitre | Ce que vous apprenez |
|---------|----------|----------------------|
| `p1_c3.py` | Chapitre 3 | Créer une API simple avec FastAPI pour servir un modèle ML |
| `p1_c4_mapping.py` | Chapitre 4 | Séparer la logique métier de l'implémentation technique |
| `p1_c4_orm.py` | Chapitre 4 | Intégrer une base de données avec SQLModel pour calculer des features historiques |

### Partie 2 — Conteneurisation, CI/CD et déploiement

| Fichier | Chapitre | Ce que vous apprenez |
|---------|----------|----------------------|
| `Dockerfile.p2_c1_1` | Chapitre 1 | Écrire un premier Dockerfile avec pip pour conteneuriser l'API |
| `Dockerfile.p2_c1_2` | Chapitre 1 | Optimiser le Dockerfile avec uv et le système de cache Docker |
| `docker-compose.yml` | Chapitre 1 | Orchestrer plusieurs services (API + PostgreSQL) |
| `.dockerignore` | Chapitre 1 | Exclure les fichiers inutiles du build Docker |
| `.github/workflows/ci.yml` | Chapitre 2 | Mettre en place l'intégration continue (lint, tests) sur les Pull Requests |
| `.github/workflows/cd.yml` | Chapitre 4 | Automatiser le déploiement continu (build et push vers AWS ECR) |
| `.github/pull_request_template.md` | Chapitre 2 | Standardiser les Pull Requests avec un template |

### Fichiers de support

| Fichier | Rôle |
|---------|------|
| `models.py` | Modèles SQLModel (définition de la table `HistoricalTransaction`) |
| `settings.py` | Configuration centralisée (base de données, MLflow, noms de colonnes) |
| `init_db.py` | Script d'initialisation de la base de données depuis un fichier parquet |

## Prérequis

Ce projet a été conçu pour fonctionner en environnement **Linux** (ou **WSL** sous Windows).

- **Python 3.12+**
- **Git**
- **Docker** (pour le déploiement conteneurisé)
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

Le projet utilise des variables d'environnement pour la configuration de la base de données et de MLflow. Ajoutez-les à votre `~/.bashrc` (ou `~/.zshrc`) :

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=ml_engineering
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=<votre_mot_de_passe>
export MLFLOW_TRACKING_URI=./mlruns
```

Puis rechargez votre shell avec `source ~/.bashrc`.

> **Note** : Lors de l'exécution via Docker, ces variables sont définies directement dans `docker-compose.yml`.

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

## Auteur

Cours conçu par **Zakaria Bouayyad** — [LinkedIn](https://www.linkedin.com/in/zakaria-bouayyad/)
