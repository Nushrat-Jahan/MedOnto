"""Étape 3 du pipeline : recherche dans SNOMED CT des concepts proposés par Qwen.

Pour chaque concept proposé à l'étape 2 (concepts_<id>.json), le script interroge l'API
BioPortal (ontologie SNOMEDCT) et garde les concepts SNOMED CT qu'elle renvoie : les candidats.
Aucun LLM n'intervient ici : chaque identifiant, libellé ou synonyme vient directement de la
réponse de l'API. Le choix du bon candidat se fera à l'étape 4.

Adapté de src/questions/recherche_snomed_B2.py (partie questionnaires) : même API, même
gestion des erreurs, mêmes informations gardées pour chaque candidat. Changements : lecture
des concepts de l'étape 2, recherche exacte avant la recherche large, hiérarchie SNOMED CT
de chaque candidat, sortie en JSON dans le dossier de la base.

La clé API est lue dans le fichier .env à la racine du projet (ligne BIOPORTAL_API_KEY=...).
Ce fichier est ignoré par git : la clé n'apparaît jamais dans le code.

Test rapide, depuis la racine du projet :
    python src/bdd/search_snomed.py "outputs/<nom_base>_<id>/concepts_<id_prompt>.json"
"""

import os
import sys
from dotenv import load_dotenv

API_URL = "https://data.bioontology.org/search"
ONTOLOGY = "SNOMEDCT"
PAGESIZE = 10  # nombre maximum de candidats gardés par concept
TIMEOUT = 30  # secondes d'attente maximum pour une réponse de l'API
MAX_RETRIES = 3  # nombre d'essais en cas d'erreur passagère (serveur surchargé, réseau)


def get_api_key() -> str:
    """Lit la clé de l'API BioPortal dans le fichier .env à la racine du projet.

    Renvoie :
        la clé, sous forme de texte.

    Lève :
        SystemExit : si la clé est absente, avec un message qui explique où la mettre.
    """
    # Les lignes du fichier .env deviennent des variables d'environnement
    load_dotenv()
    key = os.environ.get("BIOPORTAL_API_KEY", "").strip()
    if not key:
        sys.exit("Clé BIOPORTAL_API_KEY manquante : l'ajouter dans le fichier .env à la racine du projet.")
    return key
