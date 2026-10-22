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

import json
import os
import sys
import urllib.parse
import urllib.request
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


def snomed_hierarchy(synonyms: list[str]) -> str | None:
    """Retrouve la hiérarchie SNOMED CT d'un concept à partir de ses synonymes.

    Parmi les synonymes renvoyés par l'API figure le nom complet du concept, qui se termine
    par sa hiérarchie entre parenthèses, en minuscules :
    "C-reactive protein measurement (procedure)" -> "procedure".
    Une parenthèse en majuscules, comme "(ALT)", est une abréviation : elle est ignorée.

    Paramètre :
        synonyms : la liste des synonymes du concept.

    Renvoie :
        la hiérarchie, ex. "procedure" ou "observable entity" ; None si elle n'est pas trouvée.
    """
    for synonym in synonyms:
        if synonym.endswith(")"):
            start = synonym.rfind("(")
            tag = synonym[start + 1 : -1]
            if tag.islower():
                return tag
    return None


def search_once(query: str, key: str, exact: bool) -> list[dict]:
    """Envoie une recherche à l'API BioPortal et renvoie les concepts SNOMED CT trouvés.

    Deux modes de recherche :
    - exacte (exact=True) : seulement les concepts dont le nom ou un synonyme est exactement
      le terme cherché. Peu de résultats, mais fiables ;
    - large (exact=False) : les concepts qui contiennent les mots cherchés, classés par
      pertinence. Beaucoup de résultats, avec du bruit.

    Paramètres :
        query : le terme cherché, ex. "C-reactive protein measurement".
        key   : la clé de l'API.
        exact : True pour une recherche exacte, False pour une recherche large.

    Renvoie :
        au plus PAGESIZE candidats, dans l'ordre de l'API :
        {"snomed_id", "label", "hierarchy", "definition", "synonyms"}.
    """
    params = {
        "q": query,
        "ontologies": ONTOLOGY,
        "pagesize": PAGESIZE,
        "include": "prefLabel,synonym,definition",
    }
    if exact:
        params["require_exact_match"] = "true"

    # La clé est envoyée dans l'en-tête de la requête, pas dans l'adresse
    request = urllib.request.Request(
        f"{API_URL}?{urllib.parse.urlencode(params)}",
        headers={"Authorization": f"apikey token={key}"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        data = json.load(response)

    candidates = []
    seen = set()
    for concept in data.get("collection", []):
        # L'ID SNOMED CT est la fin de l'adresse du concept : .../SNOMEDCT/55235003
        snomed_id = str(concept.get("@id", "")).rsplit("/", 1)[-1]

        # On ne garde que de vrais ID SNOMED CT (des chiffres), chacun une seule fois
        if not snomed_id.isdigit() or snomed_id in seen:
            continue
        seen.add(snomed_id)

        synonyms = concept.get("synonym") or []
        definitions = concept.get("definition") or []
        definition = None
        if definitions:
            definition = definitions[0]

        candidates.append({
            "snomed_id": snomed_id,
            "label": concept.get("prefLabel"),
            "hierarchy": snomed_hierarchy(synonyms),
            "definition": definition,
            "synonyms": synonyms[:3],
        })

    return candidates
