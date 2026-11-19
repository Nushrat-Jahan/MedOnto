"""Étapes de la vérification commune : les concepts du JSON commun sont-ils dans SNOMED CT ?

Feuille de route décidée en réunion : pour chaque concept, on s'arrête dès qu'une étape le trouve.
  1. le concept lui-même est-il dans SNOMED CT ? (recherche exacte)
  2. sinon, un de ses synonymes ? (recherche exacte)
  3. sinon, Qwen choisit parmi les candidats proposés par l'API, puis regarde les enfants
  4. sinon, embedding
  5. sinon, le concept devient un candidat proposé aux experts

Les questions à l'API sont dans src/snomed_api.py.
"""

import json
from pathlib import Path

from src.snomed_api import search_exact


def verify_exact(data: dict, api_key: str) -> dict:
    """Étapes 1 et 2 de la vérification : recherche exacte du concept, puis de ses synonymes.

    Pour chaque concept du JSON commun :
    1. on cherche "concept_proposed" avec search_exact ;
    2. s'il n'est pas trouvé, on cherche ses synonymes un par un, et on s'arrête au premier
       synonyme trouvé.

    Chaque concept garde tous ses champs, et on lui ajoute le résultat de la vérification.

    Paramètres :
        data    : le contenu du JSON commun, {"<source>": [concept, ...]}.
        api_key : la clé de l'API BioPortal (renvoyée par get_api_key).

    Renvoie :
        le même format {"<source>": [concept, ...]}, où chaque concept a 4 champs en plus :
            "found_by"     : "concept", "synonym" ou "not_found" ;
            "matched_term" : le terme qui a été trouvé (le concept ou le synonyme), sinon None ;
            "snomed_id"    : l'identifiant SNOMED CT trouvé, ex. "767002", sinon None ;
            "snomed_label" : le nom officiel dans SNOMED CT, sinon None.
    """
    results = {}

    for source, concepts in data.items():
        results[source] = []

        for concept in concepts:
            # Copie du concept, avec le résultat "pas trouvé" par défaut
            result = concept.copy()
            result["found_by"] = "not_found"
            result["matched_term"] = None
            result["snomed_id"] = None
            result["snomed_label"] = None

            # 1. Le concept lui-même
            term = concept["concept_proposed"]
            found = search_exact(term, api_key)
            if found is not None:
                result["found_by"] = "concept"
                result["matched_term"] = term
                result["snomed_id"] = found["snomed_id"]
                result["snomed_label"] = found["label"]

            # 2. Sinon, ses synonymes un par un : on s'arrête au premier trouvé
            if found is None:
                for synonym in concept.get("synonyms", []):
                    found = search_exact(synonym, api_key)
                    if found is not None:
                        result["found_by"] = "synonym"
                        result["matched_term"] = synonym
                        result["snomed_id"] = found["snomed_id"]
                        result["snomed_label"] = found["label"]
                        break

            print(f"  {concept['item_id']} ({term}) : {result['found_by']}")
            results[source].append(result)

    return results


def save_verification(results: dict, concepts_path: str) -> Path:
    """Enregistre le résultat de verify_exact dans un fichier JSON, à côté du fichier des concepts.

    Le nom du fichier est celui des concepts, suivi de "_verification", ex. :
    outputs/base_a5f742cb/concepts_cc0694c8.json
    -> outputs/base_a5f742cb/concepts_cc0694c8_verification.json
    Le fichier des concepts n'est donc jamais écrasé.

    Paramètres :
        results       : le résultat de verify_exact.
        concepts_path : le chemin du fichier JSON des concepts qui a été vérifié.

    Renvoie :
        le chemin du fichier écrit.
    """
    concepts_path = Path(concepts_path)
    out_path = concepts_path.parent / (concepts_path.stem + "_verification.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"-> Vérification enregistrée : {out_path}")
    return out_path
