"""Questions à l'API BioPortal sur SNOMED CT.

Ce fichier ne contient que les fonctions qui interrogent l'API : chacune envoie une requête
et renvoie les concepts SNOMED CT trouvés (identifiant et nom officiel).
Les étapes de la vérification, qui utilisent ces fonctions, sont dans src/snomed_check.py.

La clé de l'API est lue dans le fichier .env à la racine du projet (BIOPORTAL_API_KEY=...).
"""

import json
import os
import urllib.parse
import urllib.request

from dotenv import load_dotenv

load_dotenv()  # lit le fichier .env (clé de l'API)

# Réglages de l'API BioPortal
API_URL = "https://data.bioontology.org/search"  # adresse de la recherche
ONTOLOGY = "SNOMEDCT"                            # on ne cherche que dans SNOMED CT
TIMEOUT = 30                                     # secondes d'attente maximum


def get_api_key() -> str:
    """Lit la clé de l'API BioPortal dans le fichier .env.

    Arrête le programme avec un message clair si la clé est absente.

    Renvoie :
        la clé, sous forme de texte.
    """
    key = os.environ.get("BIOPORTAL_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "BIOPORTAL_API_KEY manquante dans le fichier .env.\n"
            "Veuillez ajouter la ligne suivante dans votre .env :\n"
            "BIOPORTAL_API_KEY=votre_cle_ici"
        )
    return key


def search_exact(term: str, api_key: str) -> dict | None:
    """Cherche un terme dans SNOMED CT en recherche exacte (API BioPortal).

    La recherche exacte ne renvoie un concept que si son nom, ou un de ses synonymes, est
    exactement le terme cherché (les majuscules ne comptent pas). Contrairement à la recherche
    large, elle ne renvoie jamais un concept qui ressemble seulement au terme : un concept
    trouvé ici peut être accepté tel quel.

    On garde l'identifiant SNOMED CT du concept trouvé et son nom officiel : on peut ainsi
    vérifier le résultat, et plus tard relier la variable à SNOMED CT dans l'ontologie.

    Paramètres :
        term    : le terme à chercher, ex. "white blood cell count".
        api_key : la clé de l'API BioPortal (renvoyée par get_api_key).

    Renvoie :
        {"snomed_id": "767002", "label": "White blood cell count"} si un concept est trouvé ;
        None sinon.
    """
    # Adresse de la recherche : le terme, l'ontologie SNOMED CT, et la recherche exacte
    params = {
        "q": term,
        "ontologies": ONTOLOGY,
        "require_exact_match": "true",
        "include": "prefLabel",
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)

    # Envoi de la requête : la clé passe dans l'en-tête (elle n'apparaît pas dans l'adresse)
    request = urllib.request.Request(url, headers={"Authorization": "apikey token=" + api_key})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        data = json.load(response)

    # Premier résultat qui est un vrai concept SNOMED CT : son adresse finit par son identifiant,
    # ex. "http://purl.bioontology.org/ontology/SNOMEDCT/767002" -> "767002"
    for concept in data.get("collection", []):
        snomed_id = concept["@id"].split("/")[-1]
        if snomed_id.isdigit():
            return {"snomed_id": snomed_id, "label": concept.get("prefLabel")}

    return None


def search_broad(term: str, api_key: str, max_results: int = 10) -> list[dict]:
    """Cherche un terme dans SNOMED CT en recherche large (API BioPortal).

    La recherche large renvoie les concepts qui contiennent à peu près les mêmes mots, du plus
    au moins pertinent selon BioPortal. Ces concepts ne sont PAS forcément justes
    (ex. "date of biological test" -> "Date of positive pregnancy test") : ce sont seulement
    des candidats, que Qwen départagera ensuite.

    Paramètres :
        term        : le terme à chercher, ex. "alanine transaminase level".
        api_key     : la clé de l'API BioPortal (renvoyée par get_api_key).
        max_results : le nombre maximum de candidats renvoyés (10 par défaut).

    Renvoie :
        une liste de candidats, du plus au moins pertinent, ex. :
        [{"snomed_id": "767002", "label": "White blood cell count",
          "synonyms": ["WBC count", ...]}, ...]
        La liste est vide si rien n'est trouvé.
    """
    # Même adresse que search_exact, mais sans require_exact_match, et avec les synonymes
    params = {
        "q": term,
        "ontologies": ONTOLOGY,
        "include": "prefLabel,synonym",
        "pagesize": max_results,
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)

    # Envoi de la requête : la clé passe dans l'en-tête (elle n'apparaît pas dans l'adresse)
    request = urllib.request.Request(url, headers={"Authorization": "apikey token=" + api_key})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        data = json.load(response)

    # On garde tous les vrais concepts SNOMED CT renvoyés, dans l'ordre de BioPortal
    candidates = []
    for concept in data.get("collection", []):
        snomed_id = concept["@id"].split("/")[-1]
        if snomed_id.isdigit():
            candidates.append({
                "snomed_id": snomed_id,
                "label": concept.get("prefLabel"),
                "synonyms": concept.get("synonym", []),
            })

    return candidates
