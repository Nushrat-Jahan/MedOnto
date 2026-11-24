"""Partie questionnaires : des questions d'un questionnaire au fichier JSON commun.

Pour chaque question (item) d'un fichier de questions, Qwen propose le concept clinique
qu'elle mesure, au format JSON commun du projet. Les réponses sont nettoyées par le postprocess
commun, puis écrites dans outputs/<nom_fichier>_<id>/concepts_<id>.json, comme pour la partie bdd.

Le fichier de questions a le format de items.json :
    {"PHQ-9": {"response_type": "...", "scale": "...",
               "items": [{"item_id": "phq1", "question": "..."}, ...]}, ...}

Lancement, depuis la racine du projet, avec Ollama ouvert :
    python -m src.questions.extraction_questions "data/questionnaire/items.json"
"""

import argparse
import hashlib
import json
from pathlib import Path

import ollama

from src.postprocess import postprocess
from src.utils import load_json_file, load_txt_file, write_json_file

MODEL = "qwen2.5:7b"  # le même modèle que les autres parties
OPTIONS = {
    "temperature": 0,    # sortie stable et reproductible
    "num_predict": 400,  # longueur maximum de la réponse
}
KEEP_ALIVE = "30m"  # Qwen reste chargé en mémoire 30 min entre deux appels

PROMPT_FILE = Path("configs/systems_prompt/system_prompt.txt")  # prompt système commun
OUT_DIR = Path("outputs")  # dossier des résultats (jamais versionné)


def load_items(items_path: str | Path) -> list[dict]:
    """Lit un fichier de questions (format items.json) et renvoie la liste à plat des questions.

    Chaque question reçoit le nom de son questionnaire, son type de réponse et son échelle.
    """
    data = load_json_file(str(items_path))
    items = []
    for questionnaire, block in data.items():
        for it in block["items"]:
            items.append({
                "questionnaire": questionnaire,
                "item_id": it["item_id"],
                "question": it["question"],
                "response_type": block["response_type"],
                "scale": block["scale"],
            })
    return items


def build_user_prompt(item: dict) -> str:
    """Écrit le message envoyé à Qwen pour une question."""
    return (
        f"Questionnaire: {item['questionnaire']}\n"
        f"Item ID: {item['item_id']}\n"
        f'Item text: "{item["question"]}"\n'
        f"Response type: {item['response_type']}\n"
        f"Response scale: {item['scale']}"
    )


def ask_qwen(system_prompt: str, item: dict, user_prompt: str):
    """Un appel à Qwen pour une question. Renvoie le JSON du modèle (dict) ou None."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options=OPTIONS,
        format="json",  # force une sortie JSON valide
        keep_alive=KEEP_ALIVE,
    )
    try:
        data = json.loads(response["message"]["content"])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    data["item_id"] = item["item_id"]
    return data


def make_signature(system_prompt: str) -> str:
    """Empreinte du prompt, du modèle et des options : elle change si l'un des trois change."""
    text = system_prompt + MODEL + json.dumps(OPTIONS, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_questions(items_path: str | Path) -> Path:
    """Extraction d'un questionnaire : du fichier de questions au fichier JSON commun.

    Appelable par un autre programme (comme src/main.py) ou par la commande de ce fichier.

    Paramètre :
        items_path : le fichier de questions, ex. "data/questionnaire/items.json".

    Renvoie :
        le chemin du fichier JSON commun écrit, ex. outputs/items_<id>/concepts_<id>.json
        (c'est l'entrée de la vérification).

    Lève :
        FileNotFoundError : si le fichier n'existe pas.
    """
    items_path = Path(items_path)
    if not items_path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {items_path}")

    system_prompt = load_txt_file(str(PROMPT_FILE))
    items = load_items(items_path)
    print(f"-> {len(items)} questions lues dans '{items_path}'")

    # Un concept proposé par Qwen pour chaque question
    raw_outputs = []
    for item in items:
        print(f"-> Qwen : {item['item_id']}...")
        raw = ask_qwen(system_prompt, item, build_user_prompt(item))
        if raw is None:
            print("   ATTENTION : la réponse n'est pas un JSON valide")
        raw_outputs.append(raw)

    # Nettoyage commun : chaque question forme son propre groupe (pas de doublons entre questions)
    groups = []
    for raw in raw_outputs:
        groups.append([raw])
    cleaned = postprocess(groups)

    concepts = []
    for item, item_concepts in zip(items, cleaned):
        if not item_concepts:
            print(f"   ATTENTION {item['item_id']} : aucun concept gardé par le postprocess")
        for concept in item_concepts:
            concepts.append(concept.to_dict())
    print(f"-> Postprocess : {len(concepts)} concepts gardés sur {len(items)} questions")

    # Dossier : nom du fichier + empreinte de son contenu ; fichier : empreinte du prompt
    file_id = hashlib.sha1(items_path.read_bytes()).hexdigest()[:8]
    prompt_id = make_signature(system_prompt)[:8]
    out_dir = OUT_DIR / f"{items_path.stem.replace(' ', '_')}_{file_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    concepts_path = out_dir / f"concepts_{prompt_id}.json"

    # Format JSON commun, comme la partie bdd : {"<source>": [concept, ...]}
    write_json_file(str(concepts_path), {str(items_path): concepts})
    print(f"-> {len(concepts)} concepts au format commun dans '{concepts_path}'")

    return concepts_path


def main() -> None:
    """Commande : lit le chemin du fichier de questions dans le terminal, puis lance run_questions."""
    parser = argparse.ArgumentParser(description="Extraction d'un questionnaire -> JSON commun")
    parser.add_argument("items", help="fichier de questions au format items.json")
    args = parser.parse_args()

    try:
        run_questions(args.items)
    except FileNotFoundError as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
