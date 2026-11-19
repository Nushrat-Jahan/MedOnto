"""Étapes de la vérification commune : les concepts du JSON commun sont-ils dans SNOMED CT ?

Feuille de route décidée en réunion : pour chaque concept, on s'arrête dès qu'une étape le trouve.
  1. le concept lui-même est-il dans SNOMED CT ? (recherche exacte)
  2. sinon, un de ses synonymes ? (recherche exacte)
  3. sinon, Qwen choisit parmi les candidats proposés par l'API, puis regarde les enfants
  4. sinon, embedding
  5. sinon, le concept devient un candidat proposé aux experts

Les questions à l'API sont dans src/verification/snomed_api.py, celles posées à Qwen dans
src/verification/llm_choice.py.
"""

import json
from pathlib import Path

from src.verification.llm_choice import ask_qwen_choice, ask_qwen_new_term
from src.verification.snomed_api import get_direct_children, search_broad, search_exact

MAX_NEW_TERMS = 2  # étape 3b : nombre de nouveaux noms demandés à Qwen, au maximum


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


def check_with_llm(concept: dict, api_key: str) -> dict:
    """Étape 3 de la vérification, pour UN concept que la recherche exacte n'a pas trouvé.

    3a. L'API propose des candidats (recherche large) et Qwen choisit le plus proche.
    3b. Si Qwen n'en choisit aucun, il propose un autre nom (MAX_NEW_TERMS essais au maximum) :
        on cherche ce nom en recherche exacte, sinon en recherche large avec le choix de Qwen.
    3c. Si le concept trouvé est plus général ("broader"), Qwen regarde ses enfants directs :
        un enfant de même sens ("exact") remplace le parent.

    Qwen ne choisit que parmi de vrais concepts SNOMED CT renvoyés par l'API. Un concept trouvé
    ici n'est pas sûr pour autant : il devra être confirmé par les experts.

    Paramètres :
        concept : un concept du JSON commun (on utilise "concept_proposed", "evidence", "synonyms").
        api_key : la clé de l'API BioPortal (renvoyée par get_api_key).

    Renvoie :
        {"found_by": "llm_choice", "matched_term": "...", "snomed_id": "69480007",
         "snomed_label": "...", "llm_match": "exact", "llm_reason": "<justification de Qwen>"}
        found_by vaut "llm_choice" (3a), "llm_new_term" (3b), "llm_child" (3c) ou "not_found".
    """
    result = {
        "found_by": "not_found",
        "matched_term": None,
        "snomed_id": None,
        "snomed_label": None,
        "llm_match": None,
        "llm_reason": None,
    }
    term = concept["concept_proposed"]

    # Termes déjà essayés, donnés à Qwen pour qu'il ne les repropose pas
    tried_terms = [term]
    for synonym in concept.get("synonyms", []):
        tried_terms.append(synonym)

    # 3a. L'API propose des candidats, Qwen choisit
    candidates = search_broad(term, api_key)
    choice = ask_qwen_choice(concept, candidates)
    result["llm_reason"] = choice["reason"]
    if choice["snomed_id"] is not None:
        result["found_by"] = "llm_choice"
        result["matched_term"] = term
        result["snomed_id"] = choice["snomed_id"]
        result["snomed_label"] = choice["label"]
        result["llm_match"] = choice["match"]

    # 3b. Sinon, Qwen propose un autre nom, au plus MAX_NEW_TERMS fois
    if result["snomed_id"] is None:
        for candidate in candidates:
            tried_terms.append(candidate["label"])

        attempt = 1
        while attempt <= MAX_NEW_TERMS and result["snomed_id"] is None:
            answer = ask_qwen_new_term(concept, tried_terms)
            new_term = answer["term"]
            if new_term is None:  # réponse illisible ou terme répété : on arrête
                break
            tried_terms.append(new_term)

            # Le nouveau nom existe-t-il tel quel dans SNOMED CT ?
            found = search_exact(new_term, api_key)
            if found is not None:
                result["found_by"] = "llm_new_term"
                result["matched_term"] = new_term
                result["snomed_id"] = found["snomed_id"]
                result["snomed_label"] = found["label"]
                result["llm_match"] = answer["match"]
                result["llm_reason"] = "name proposed by Qwen, found by exact search"
            else:
                # Sinon : recherche large sur ce nom, et Qwen choisit parmi les candidats
                new_candidates = search_broad(new_term, api_key)
                choice = ask_qwen_choice(concept, new_candidates)
                if choice["snomed_id"] is not None:
                    result["found_by"] = "llm_new_term"
                    result["matched_term"] = new_term
                    result["snomed_id"] = choice["snomed_id"]
                    result["snomed_label"] = choice["label"]
                    result["llm_match"] = choice["match"]
                    result["llm_reason"] = choice["reason"]
                else:
                    for candidate in new_candidates:
                        tried_terms.append(candidate["label"])

            attempt = attempt + 1

    # 3c. Si le concept trouvé est plus général, on regarde ses enfants directs
    if result["snomed_id"] is not None and result["llm_match"] == "broader":
        children = get_direct_children(result["snomed_id"], api_key)
        child_choice = ask_qwen_choice(concept, children)
        if child_choice["snomed_id"] is not None and child_choice["match"] == "exact":
            result["found_by"] = "llm_child"
            result["snomed_id"] = child_choice["snomed_id"]
            result["snomed_label"] = child_choice["label"]
            result["llm_match"] = "exact"
            result["llm_reason"] = child_choice["reason"]

    return result
