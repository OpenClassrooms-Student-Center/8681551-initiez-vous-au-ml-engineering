# Cours ML Engineering - Initiez-vous au ML Engineering

Ce dépôt accompagne le cours OpenClassrooms **"Initiez-vous au ML Engineering"** destiné aux data scientists souhaitant évoluer vers des rôles de ML Engineering.

**Lien du cours** : https://openclassrooms.com/fr/courses/8681551-initiez-vous-au-ml-engineering

## Table des matières

- [À propos de ce projet](#à-propos-de-ce-projet)
- [Objectifs pédagogiques](#objectifs-pédagogiques)
- [Prérequis](#prérequis)
- [Démarrage rapide](#démarrage-rapide)
- [Variables d'environnement](#variables-denvironnement)
- [Structure du projet](#structure-du-projet)
- [Architecture](#architecture)
- [Référence API](#référence-api)
- [Schéma de base de données](#schéma-de-base-de-données)
- [Notes sur la qualité du code](#notes-sur-la-qualité-du-code)
- [Problèmes courants et solutions](#problèmes-courants-et-solutions)
- [Ressources](#ressources)

## À propos de ce projet

Ce projet de formation au ML Engineering démontre des patterns de production avec l'apprentissage supervisé. Il prédit les catégories de prix immobiliers français (au-dessus/en-dessous du marché) en utilisant CatBoost, FastAPI, SQLModel et MLflow.

Le code est organisé pour suivre votre parcours d'apprentissage :

1. **Commencez ici** : `p1_c3.py` - Version la plus simple, un endpoint FastAPI basique qui charge un modèle et fait des prédictions. Pas de base de données, complexité minimale.

2. **Ensuite** : `p1_c4_mapping.py` - Introduit le concept de séparation entre la logique métier (entrées conviviales) et l'implémentation technique (entrées du modèle ML).

3. **Avancé** : `p1_c4_orm.py` - Version production complète avec intégration base de données, calcul de features historiques et gestion d'erreurs complète.

## Objectifs pédagogiques

Ce code vous apprend à :

- **Servir des modèles ML** via des APIs REST avec FastAPI
- **Suivre les expériences** et gérer les versions de modèles avec MLflow
- **Conteneuriser des applications** avec Docker pour des déploiements reproductibles
- **Se connecter aux bases de données** avec SQLModel (combine SQLAlchemy + Pydantic)
- **Structurer du code production** avec une séparation claire entre couches métier et techniques

## Prérequis

Avant de commencer, assurez-vous d'avoir installé :

- **Python 3.12+**
- **Docker Desktop** (pour le déploiement conteneurisé)
- **Git** pour le contrôle de version
- **uv** (optionnel, mais recommandé) - gestionnaire de paquets Python rapide

### Installation de uv

```bash
# macOS / Linux / WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

## Démarrage rapide

### Option 1 : Exécution avec Docker (Recommandé)

C'est la méthode la plus simple - tout est préconfiguré :

```bash
# Cloner le dépôt
git clone <repo-url>
cd cours_ml_engineering

# Démarrer tout (API + base de données PostgreSQL)
docker-compose up --build

# L'API est maintenant accessible à http://localhost:8000
# Ouvrez http://localhost:8000/docs dans votre navigateur pour voir la documentation interactive
```

### Option 2 : Exécution locale avec uv

```bash
# Créer l'environnement virtuel et installer les dépendances
uv sync

# Démarrer PostgreSQL (nécessaire pour p1_c4_orm.py)
docker-compose up postgres -d

# Initialiser la base de données
python init_db.py

# Démarrer le serveur de développement
uv run uvicorn p1_c4_orm:app --reload

# L'API est maintenant accessible à http://localhost:8000
```

### Tester votre installation

Une fois le serveur démarré :

1. **Visitez la documentation interactive** : http://localhost:8000/docs
2. **Vérifiez la santé de l'API** : http://localhost:8000/health
3. **Essayez une prédiction** : Utilisez l'endpoint `/predict/classification` dans Swagger UI

## Variables d'environnement

### Configuration de la base de données

| Variable | Description | Valeur par défaut |
|----------|-------------|-------------------|
| `POSTGRES_HOST` | Hôte PostgreSQL | `localhost` |
| `POSTGRES_PORT` | Port PostgreSQL | `5432` |
| `POSTGRES_DB` | Nom de la base de données | `ml_engineering` |
| `POSTGRES_USER` | Utilisateur PostgreSQL | `postgres` |
| `POSTGRES_PASSWORD` | Mot de passe PostgreSQL | `password` |

### Configuration MLflow

| Variable | Description | Valeur par défaut |
|----------|-------------|-------------------|
| `MLFLOW_TRACKING_URI` | URI de suivi MLflow | `./mlruns` (chemin local) |
| `MLFLOW_REMOTE_TRACKING_URI` | URI du serveur MLflow distant | (serveur EC2) |

### Configuration par système d'exploitation

#### macOS / Linux

```bash
# Fichier .env ou export direct
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=ml_engineering
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=votre_mot_de_passe_securise
export MLFLOW_TRACKING_URI=./mlruns
```

#### WSL (Windows Subsystem for Linux)

```bash
# Dans WSL, utilisez les mêmes commandes que Linux
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=ml_engineering
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=votre_mot_de_passe_securise
export MLFLOW_TRACKING_URI=./mlruns

# Si PostgreSQL tourne sur Windows (pas dans WSL), utilisez l'IP Windows :
export POSTGRES_HOST=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
```

#### Fichier .env (recommandé)

Créez un fichier `.env` à la racine du projet :

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=ml_engineering
POSTGRES_USER=postgres
POSTGRES_PASSWORD=votre_mot_de_passe_securise
MLFLOW_TRACKING_URI=./mlruns
```

> **Note de sécurité** : Ne commitez jamais de fichiers `.env` contenant des mots de passe réels. Ajoutez `.env` à votre `.gitignore`.

## Structure du projet

```
cours_ml_engineering/
├── p1_c3.py                 # API simple (débutant)
├── p1_c4_mapping.py         # API avec mapping métier/technique (intermédiaire)
├── p1_c4_orm.py             # API complète avec ORM (avancé)
├── models.py                # Modèles SQLModel (HistoricalTransaction)
├── settings.py              # Configuration (DB, MLflow, colonnes)
├── init_db.py               # Script d'initialisation de la base de données
├── docker-compose.yml       # Orchestration Docker (API + PostgreSQL)
├── Dockerfile.p2_c1_2       # Dockerfile de production (optimisé UV)
├── pyproject.toml           # Configuration du projet Python
├── requirements.txt         # Dépendances (généré depuis pyproject.toml)
├── uv.lock                  # Fichier de verrouillage des dépendances
├── mlruns/                  # Répertoire de suivi MLflow (local)
├── tests/                   # Répertoire des tests
└── .github/workflows/       # CI/CD GitHub Actions
    ├── ci.yml               # Intégration continue
    └── cd.yml               # Déploiement continu
```

### Fichiers clés

| Fichier | Objectif | Difficulté |
|---------|----------|------------|
| `p1_c3.py` | Version débutant - API simple sans base de données | Débutant |
| `p1_c4_mapping.py` | Couche de mapping features métier/technique | Intermédiaire |
| `p1_c4_orm.py` | **API principale de production** - FastAPI complet avec ORM | Avancé |
| `models.py` | Modèles de base de données SQLModel (HistoricalTransaction) | Intermédiaire |
| `settings.py` | Configuration (DB, URIs MLflow, noms de colonnes) | Débutant |
| `init_db.py` | Initialisation de la base de données depuis parquet | Intermédiaire |
| `Dockerfile.p2_c1_2` | Dockerfile de production (optimisé UV) | Intermédiaire |

## Architecture

### Couches applicatives

```
Endpoints FastAPI (p1_c4_orm.py)
         │
         ▼
Couche Logique Métier (p1_c4_mapping.py)
  - Mapping PropertyFeaturesBusiness → PropertyFeatures
  - Calcul des features historiques depuis la DB
         │
         ▼
Couche Données/Modèle
  - ORM SQLModel (models.py)
  - Chargement de modèle MLflow
  - Base de données PostgreSQL
```

### Diagramme de flux

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Client HTTP   │────▶│   FastAPI App    │────▶│   CatBoost      │
│   (Swagger UI)  │     │   (p1_c4_orm)    │     │   Classifier    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                         │
                               ▼                         │
                        ┌──────────────────┐            │
                        │   PostgreSQL     │◀───────────┘
                        │   (Historique)   │   Features historiques
                        └──────────────────┘
                               ▲
                               │
                        ┌──────────────────┐
                        │     MLflow       │
                        │   (Modèles)      │
                        └──────────────────┘
```

## Référence API

### Endpoints disponibles (p1_c4_orm.py)

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/` | Métadonnées de l'API |
| `GET` | `/health` | Vérification de santé du modèle et de l'API |
| `POST` | `/predict/classification` | Endpoint principal de prédiction |

### Exemple de requête

```bash
curl -X POST "http://localhost:8000/predict/classification" \
  -H "Content-Type: application/json" \
  -d '{
    "Type de Bâtiment": "Appartement",
    "Surface Habitable (m²)": 75.5,
    "Longitude": -0.5792,
    "Latitude": 44.8378,
    "Plan VEFA": "Non",
    "Année de Transaction": 2023,
    "Mois de Transaction": 6,
    "Taux d'\''Intérêt (%)": 3.2
  }'
```

### Exemple de réponse

```json
{
  "prediction": 1,
  "probability": 0.85,
  "model_run_id": "abc123def456"
}
```

### Signification des prédictions

| Valeur | Signification |
|--------|---------------|
| `0` | Prix en dessous du marché |
| `1` | Prix au-dessus du marché |

## Schéma de base de données

### Table `historicaltransaction`

La table `HistoricalTransaction` contient plus de 50 features issues des transactions immobilières de Nouvelle-Aquitaine.

#### Colonnes principales

| Colonne | Type | Description |
|---------|------|-------------|
| `id_transaction` | `INT` (PK) | Identifiant unique de la transaction |
| `prix` | `FLOAT` | Prix de la transaction |
| `surface_habitable` | `INT` | Surface habitable en m² |
| `latitude` | `FLOAT` | Latitude du bien |
| `longitude` | `FLOAT` | Longitude du bien |
| `mois_transaction` | `INT` | Mois de la transaction (1-12) |
| `annee_transaction` | `INT` | Année de la transaction |
| `en_dessous_du_marche` | `INT` | Target : 0 = en dessous, 1 = au-dessus |
| `type_batiment_Appartement` | `INT` | 1 si appartement, 0 sinon |
| `vefa` | `INT` | 1 si VEFA, 0 sinon |

#### Features économiques

| Colonne | Type | Description |
|---------|------|-------------|
| `prix_m2_moyen_mois_precedent` | `FLOAT` | Prix moyen au m² du mois précédent |
| `nb_transactions_mois_precedent` | `INT` | Nombre de transactions du mois précédent |
| `taux_interet` | `FLOAT` | Taux d'intérêt |
| `variation_taux_interet` | `FLOAT` | Variation du taux d'intérêt |
| `acceleration_taux_interet` | `FLOAT` | Accélération du taux d'intérêt |

## Commandes utiles

### Développement

```bash
uv sync                                    # Installer les dépendances
uv run uvicorn p1_c4_orm:app --reload     # Lancer le serveur de dev (port 8000)
uv run pytest --cov --cov-report=term-missing  # Lancer les tests avec couverture
```

### Qualité du code

```bash
ruff check --fix     # Linting avec corrections automatiques
ruff format --check  # Vérifier le formatage
```

### Docker

```bash
docker-compose up --build  # Démarrer app + PostgreSQL
docker-compose down        # Arrêter tous les services
docker-compose logs -f     # Voir les logs en temps réel
```

### Base de données

```bash
python init_db.py              # Initialiser avec les données historiques
python init_db.py --drop-existing  # Réinitialiser et recharger les données
```

## Notes sur la qualité du code

### Points d'amélioration identifiés

Ce code est conçu à des fins pédagogiques. Voici quelques améliorations possibles pour un environnement de production :

#### Sécurité

- **Gestion des secrets** : Les mots de passe par défaut dans `settings.py` et `docker-compose.yml` devraient utiliser des gestionnaires de secrets (AWS Secrets Manager, HashiCorp Vault, etc.)

#### Performance

- **Insertion en base de données** : Le script `init_db.py` insère les données ligne par ligne. Pour de grands volumes, utilisez `session.add_all()` ou des insertions bulk
- **Wrapper SQL** : L'appel `session.exec("TRUNCATE...")` devrait utiliser `text()` de SQLAlchemy

#### Maintenabilité

- **Chemins de fichiers** : Le chemin du fichier parquet dans `init_db.py` est codé en dur
- **Code répétitif** : Le calcul des moyennes dans `p1_c4_orm.py` est répété 4 fois et pourrait être refactorisé

## Problèmes courants et solutions

### Le serveur ne démarre pas

**Symptôme** : Erreur de connexion à la base de données

**Solution** :
```bash
# Vérifiez que PostgreSQL est démarré
docker-compose up postgres -d

# Vérifiez la connexion
docker-compose logs postgres
```

### Erreur "No MLflow runs found"

**Symptôme** : L'API ne trouve pas de modèle MLflow

**Solution** :
- Vérifiez que le répertoire `mlruns/` existe et contient des expériences
- Vérifiez la variable `MLFLOW_TRACKING_URI`

### Erreur "Aucune donnée historique"

**Symptôme** : L'endpoint de prédiction retourne une erreur 400

**Solution** :
```bash
# Initialisez la base de données
python init_db.py

# Ou réinitialisez si nécessaire
python init_db.py --drop-existing
```

### Erreur de port déjà utilisé

**Symptôme** : `Error: port 8000 already in use`

**Solution** :
```bash
# Trouvez le processus utilisant le port
lsof -i :8000

# Tuez le processus
kill -9 <PID>

# Ou utilisez un autre port
uvicorn p1_c4_orm:app --port 8001
```

### WSL : Connexion à PostgreSQL Windows

**Symptôme** : Impossible de se connecter à PostgreSQL depuis WSL

**Solution** :
```bash
# Utilisez l'IP de l'hôte Windows
export POSTGRES_HOST=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
```

## Ressources

### Documentation officielle

- [FastAPI](https://fastapi.tiangolo.com/)
- [SQLModel](https://sqlmodel.tiangolo.com/)
- [MLflow](https://mlflow.org/docs/latest/index.html)
- [CatBoost](https://catboost.ai/docs/)
- [Pydantic](https://docs.pydantic.dev/)

### Cours OpenClassrooms

- [Initiez-vous au ML Engineering](https://openclassrooms.com/fr/courses/8681551-initiez-vous-au-ml-engineering)

### Outils de développement

- [uv - Gestionnaire de paquets Python](https://docs.astral.sh/uv/)
- [Ruff - Linter Python](https://docs.astral.sh/ruff/)
- [Docker](https://docs.docker.com/)

## CI/CD

- **CI** (Pull Requests) : Lint Ruff, formatage, tests pytest avec couverture
- **CD** (branche main) : Build et push vers AWS ECR

## Stack technique

- **ML** : CatBoost, scikit-learn, MLflow
- **API** : FastAPI, uvicorn
- **DB** : PostgreSQL, SQLModel
- **Data** : pandas, polars, duckdb
- **Dev** : pytest, ruff, uv

## Licence

MIT
