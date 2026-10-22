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
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
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


def search_candidates(query: str, key: str, exact: bool) -> list[dict]:
    """Lance search_once, en réessayant en cas d'erreur passagère.

    Si le serveur est surchargé (erreur 429, « trop de requêtes ») ou en panne (erreurs 500,
    502, 503), ou si le réseau coupe, on attend puis on réessaie : 5 s, puis 10 s, puis on
    abandonne. Une autre erreur (ex. 401 : clé refusée) arrête tout de suite les essais.

    Paramètres :
        query, key, exact : comme pour search_once.

    Renvoie :
        les candidats trouvés ; une liste vide si la recherche a échoué (message affiché).
    """
    for attempt in range(MAX_RETRIES):
        try:
            return search_once(query, key, exact)
        except urllib.error.HTTPError as error:
            # Erreur passagère du serveur : on attend et on réessaie
            if error.code in (429, 500, 502, 503) and attempt < MAX_RETRIES - 1:
                time.sleep(5 * (attempt + 1))
                continue
            print(f"   ERREUR HTTP {error.code} pour la recherche '{query}'")
            return []
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            # Problème de réseau ou réponse illisible : on attend et on réessaie
            if attempt < MAX_RETRIES - 1:
                time.sleep(5 * (attempt + 1))
                continue
            print(f"   ERREUR pour la recherche '{query}' : {error}")
            return []
    return []


def search_concepts(concepts: list[dict], key: str) -> list[dict]:
    """Cherche dans SNOMED CT le concept proposé par Qwen pour chaque variable.

    Stratégie, pour chaque concept :
    1. recherche exacte du terme proposé ;
    2. si elle ne trouve rien, recherche large (au plus PAGESIZE candidats) ;
    3. si toujours rien : aucun candidat (la variable deviendra un Candidate Project Concept).

    Les "_" que Qwen met parfois entre les mots sont remplacés par des espaces
    ("alanine_transaminase_measurement" -> "alanine transaminase measurement").
    Les candidats ne sont pas filtrés par catégorie : celle proposée par Qwen n'est pas fiable
    (ex. une mesure de laboratoire classée observable_entity au lieu de procedure). La
    hiérarchie de chaque candidat est gardée, pour le choix de l'étape 4.

    Paramètres :
        concepts : la liste "concepts" de concepts_<id>.json (renvoyée par extract_concepts).
        key      : la clé de l'API.

    Renvoie :
        une liste avec un dictionnaire par variable :
        {"variable", "query", "search_mode", "candidates"}, où search_mode vaut "exact",
        "broad" ou None (aucun candidat, ou pas de concept proposé par Qwen).
    """
    searches = []

    for item in concepts:
        concept = item["concept"]

        # Réponse invalide de Qwen : rien à chercher
        if concept is None or not concept.get("concept_proposed"):
            searches.append({"variable": item["variable"], "query": None,
                             "search_mode": None, "candidates": []})
            continue

        query = str(concept["concept_proposed"]).replace("_", " ")
        print(f"-> SNOMED CT : {item['variable']} ('{query}')...")

        # 1. Recherche exacte, 2. sinon recherche large
        search_mode = "exact"
        candidates = search_candidates(query, key, exact=True)
        if not candidates:
            search_mode = "broad"
            candidates = search_candidates(query, key, exact=False)
        if not candidates:
            search_mode = None

        print(f"   {len(candidates)} candidat(s), recherche {search_mode}")
        searches.append({
            "variable": item["variable"],
            "query": query,
            "search_mode": search_mode,
            "candidates": candidates,
        })

    return searches


def save_candidates(searches: list[dict], concepts_path: str | Path) -> Path:
    """Enregistre les candidats SNOMED CT à côté du fichier de concepts dont ils viennent.

    Le nom reprend celui du fichier de concepts : concepts_afaf0d45.json donne
    candidates_afaf0d45.json. On sait ainsi toujours de quels concepts viennent les candidats.

    Paramètres :
        searches      : la liste renvoyée par search_concepts.
        concepts_path : le chemin du fichier concepts_<id>.json lu en entrée.

    Renvoie :
        le chemin du fichier écrit.
    """
    concepts_path = Path(concepts_path)

    # Même identifiant que le fichier de concepts (jamais le même nom : il n'est pas écrasé)
    out_name = "candidates_" + concepts_path.name.removeprefix("concepts_")
    out_path = concepts_path.parent / out_name

    # Contenu : en-tête (fichier de concepts, ontologie) puis les recherches
    output = {
        "concepts_file": concepts_path.name,
        "ontology": ONTOLOGY,
        "searches": searches,
    }
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    # Test manuel : lit un concepts_<id>.json produit par l'étape 2, puis interroge l'API
    if len(sys.argv) != 2:
        sys.exit("Usage : python src/bdd/search_snomed.py <chemin_de_concepts.json>")
    concepts_path = Path(sys.argv[1])
    data = json.loads(concepts_path.read_text(encoding="utf-8"))

    searches = search_concepts(data["concepts"], get_api_key())

    # Une ligne par variable : terme cherché, mode de recherche, premier candidat
    print()
    for search in searches:
        first = "aucun candidat"
        if search["candidates"]:
            best = search["candidates"][0]
            first = f"{best['snomed_id']} {best['label']} ({best['hierarchy']})"
        print(f"{search['variable']:15} | {search['query']} | {search['search_mode']} | {first}")

    out_path = save_candidates(searches, concepts_path)
    print(f"\n{len(searches)} recherches enregistrées dans '{out_path}'")
