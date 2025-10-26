"""
Database Models

This module contains all SQLModel database models shared across the application.
"""

from sqlmodel import SQLModel, Field as SQLField
from typing import Optional

# Import SQLModel to register metadata
__all__ = ["HistoricalTransaction"]


class HistoricalTransaction(SQLModel, table=True):
    """Modèle SQLModel pour les transactions historiques de Nouvelle-Aquitaine avec PostgreSQL"""

    id_transaction: int = SQLField(primary_key=True)
    prix: float
    type_batiment_Appartement: Optional[int] = None
    type_batiment_Maison: Optional[int] = None
    vefa: int
    n_pieces: int
    surface_habitable: int
    latitude: float
    longitude: float
    mois_transaction: int
    annee_transaction: int
    en_dessous_du_marche: int
    nom_region_Auvergne_Rhone_Alpes: Optional[int] = None
    nom_region_Nouvelle_Aquitaine: Optional[int] = None
    nom_region_Occitanie: Optional[int] = None
    nom_region_Provence_Alpes_Cote_d_Azur: Optional[int] = None
    nom_region_Ile_de_France: Optional[int] = None
    ville_demandee: Optional[int] = None
    prix_m2_moyen_mois_precedent: Optional[float] = None
    nb_transactions_mois_precedent: Optional[int] = None
    prix_m2_moyen_glissant_6mo: Optional[float] = None
    nb_transaction_moyen_glissant_6mo: Optional[float] = None
    taux_endettement: Optional[float] = None
    euros_par_habitant: Optional[float] = None
    fraction_depot_banque: Optional[float] = None
    fraction_assurance_vie: Optional[float] = None
    fraction_fonds_communs: Optional[float] = None
    fraction_fond_pension: Optional[float] = None
    fraction_titres_non_action: Optional[float] = None
    fraction_actions: Optional[float] = None
    variation_taux_endettement: Optional[float] = None
    acceleration_taux_endettement: Optional[float] = None
    taux_interet: Optional[float] = None
    emprunts_ME: Optional[float] = None
    indice_reference_loyers: Optional[float] = None
    variation_taux_interet: Optional[float] = None
    acceleration_taux_interet: Optional[float] = None
    moyenne_glissante_6mo_variation_taux_interet: Optional[float] = None
