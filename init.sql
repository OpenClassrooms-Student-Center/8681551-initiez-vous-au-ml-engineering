-- Script d'initialisation PostgreSQL pour ML Engineering
-- Ce script est exécuté automatiquement lors du premier démarrage du conteneur

-- Créer la base de données si elle n'existe pas déjà
-- (La base est déjà créée via la variable d'environnement POSTGRES_DB)

-- Créer un utilisateur dédié pour l'application
CREATE USER ml_user WITH PASSWORD 'ml_password';
GRANT ALL PRIVILEGES ON DATABASE ml_engineering TO ml_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ml_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ml_user;

-- Configurer les extensions utiles
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Afficher les informations de connexion
SELECT 'Base de données ml_engineering initialisée avec succès' as status;
