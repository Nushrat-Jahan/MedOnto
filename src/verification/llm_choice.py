"""Questions posées à Qwen pendant la vérification (étape 3).

Règle : Qwen propose, l'API vérifie. Qwen ne choisit que parmi de vrais concepts SNOMED CT
renvoyés par l'API : il ne peut donc pas inventer de concept ni d'identifiant.

Qwen tourne en local via Ollama (le logiciel Ollama doit être lancé), à temperature=0 :
la même question donne toujours la même réponse.
"""

import json
import sys
from pathlib import Path

import ollama

MODEL = "qwen2.5:7b"  # le même modèle que pour l'extraction

# Prompt du choix : {concept}, {evidence} et {candidates} sont remplacés pour chaque concept.
# Chemin relatif : les scripts se lancent depuis la racine du projet.
CHOICE_PROMPT_PATH = Path("configs/verification_prompts/choice_prompt.txt")
CHOICE_PROMPT = CHOICE_PROMPT_PATH.read_text(encoding="utf-8")
for placeholder in ["{concept}", "{evidence}", "{candidates}"]:
    if placeholder not in CHOICE_PROMPT:
        sys.exit(f"{CHOICE_PROMPT_PATH} doit contenir le repère {placeholder}")

# Prompt du nouveau terme : {concept}, {evidence} et {tried} sont remplacés pour chaque concept.
NEW_TERM_PROMPT_PATH = Path("configs/verification_prompts/new_term_prompt.txt")
NEW_TERM_PROMPT = NEW_TERM_PROMPT_PATH.read_text(encoding="utf-8")
for placeholder in ["{concept}", "{evidence}", "{tried}"]:
    if placeholder not in NEW_TERM_PROMPT:
        sys.exit(f"{NEW_TERM_PROMPT_PATH} doit contenir le repère {placeholder}")


def ask_qwen_choice(concept: dict, candidates: list[dict]) -> dict:
    """Demande à Qwen de choisir, parmi les candidats de l'API, le concept SNOMED CT le plus proche.

    Qwen répond par le numéro du candidat choisi (0 s'il n'y en a aucun de correct), le type de
    correspondance ("exact" : même sens ; "broader" : plus général mais correct) et une courte
    justification. Le script vérifie que le numéro est bien dans la liste : un numéro hors liste
    ou une réponse illisible comptent comme "aucun".

    Paramètres :
        concept    : un concept du JSON commun (on utilise "concept_proposed" et "evidence").
        candidates : les candidats renvoyés par search_broad.

    Renvoie :
        {"snomed_id": "69480007", "label": "Gamma glutamyl transferase measurement",
         "match": "exact", "reason": "<justification de Qwen>", "raw_answer": "<réponse brute>"}
        snomed_id, label et match valent None si Qwen n'a choisi aucun candidat.
    """
    result = {
        "snomed_id": None,
        "label": None,
        "match": None,
        "reason": None,
        "raw_answer": None,
    }

    # Pas de candidats : rien à choisir, on ne demande pas à Qwen
    if not candidates:
        result["reason"] = "no candidate returned by the API"
        return result

    # 1. Liste numérotée des candidats, avec leurs synonymes
    lines = []
    number = 1
    for candidate in candidates:
        line = f"{number}. {candidate['label']}"
        if candidate["synonyms"]:
            line = line + " (synonyms: " + "; ".join(candidate["synonyms"]) + ")"
        lines.append(line)
        number = number + 1

    prompt = CHOICE_PROMPT.replace("{concept}", concept["concept_proposed"])
    prompt = prompt.replace("{evidence}", str(concept.get("evidence", "")))
    prompt = prompt.replace("{candidates}", "\n".join(lines))

    # 2. Question à Qwen : réponse en JSON, temperature=0
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
        format="json",
    )
    raw = response["message"]["content"]
    result["raw_answer"] = raw

    # 3. Lecture de la réponse : une réponse illisible compte comme "aucun"
    try:
        answer = json.loads(raw)
    except json.JSONDecodeError:
        return result
    if not isinstance(answer, dict):
        return result

    result["reason"] = answer.get("reason")

    # Le numéro peut arriver sous forme de texte ("3") : on le transforme en nombre
    choice = answer.get("choice")
    if isinstance(choice, str) and choice.isdigit():
        choice = int(choice)

    # On n'accepte qu'un numéro de la liste (de 1 au nombre de candidats) ; 0 = aucun
    if isinstance(choice, int) and 1 <= choice <= len(candidates):
        chosen = candidates[choice - 1]
        result["snomed_id"] = chosen["snomed_id"]
        result["label"] = chosen["label"]
        result["match"] = answer.get("match")

    return result


def ask_qwen_new_term(concept: dict, tried_terms: list[str]) -> dict:
    """Demande à Qwen un autre nom pour un concept qui n'a pas été trouvé dans SNOMED CT.

    Sert quand Qwen n'a choisi aucun des candidats de l'API (ask_qwen_choice). On lui donne
    les termes déjà essayés, et il propose un seul nouveau nom, écrit comme dans SNOMED CT.
    Ce nom n'est PAS vérifié ici : il faut ensuite le chercher avec search_exact.

    Paramètres :
        concept     : un concept du JSON commun (on utilise "concept_proposed" et "evidence").
        tried_terms : les termes déjà essayés (le concept, ses synonymes, les candidats rejetés).

    Renvoie :
        {"term": "laboratory test date", "match": "exact", "raw_answer": "<réponse brute>"}
        term et match valent None si la réponse est illisible ou répète un terme déjà essayé.
    """
    result = {
        "term": None,
        "match": None,
        "raw_answer": None,
    }

    # 1. Liste des termes déjà essayés, un par ligne
    lines = []
    for tried in tried_terms:
        lines.append("- " + tried)

    prompt = NEW_TERM_PROMPT.replace("{concept}", concept["concept_proposed"])
    prompt = prompt.replace("{evidence}", str(concept.get("evidence", "")))
    prompt = prompt.replace("{tried}", "\n".join(lines))

    # 2. Question à Qwen : réponse en JSON, temperature=0
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
        format="json",
    )
    raw = response["message"]["content"]
    result["raw_answer"] = raw

    # 3. Lecture de la réponse : une réponse illisible ne donne aucun terme
    try:
        answer = json.loads(raw)
    except json.JSONDecodeError:
        return result
    if not isinstance(answer, dict):
        return result

    # On n'accepte qu'un texte non vide
    term = answer.get("term")
    if not isinstance(term, str) or term.strip() == "":
        return result
    term = term.strip()

    # On refuse un terme déjà essayé (les majuscules ne comptent pas)
    for tried in tried_terms:
        if term.lower() == tried.lower():
            return result

    result["term"] = term
    result["match"] = answer.get("match")
    return result
