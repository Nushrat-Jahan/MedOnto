
"""
Extraction des concepts cliniques des items de questionnaires (PHQ-9, GAD-7, ESS)
avec Qwen2.5-7B-Instruct (Ollama), en utilisant le prompt et le post-traitement
communs au groupe.

Entrées :
  - configs/systems_prompt/system_prompt.txt   
  - src/questions/items.json               
Sorties (dans res/questions/) :
  - concepts_questions.json   (item_id, concept_proposed, category, relation,
                               evidence, synonyms)
  - concepts_questions.csv    (mêmes champs, à plat)

"""

import csv
import hashlib
import json
import time
from pathlib import Path
 
import ollama
 
from src.postprocess import postprocess
from src.utils import load_json_file, load_txt_file, write_json_file
 
MODEL = "qwen2.5:7b-instruct"
OPTIONS = {
    "temperature": 0,    # sortie stable et reproductible
    "num_predict": 400,  # plafond de longueur de la réponse (le JSON attendu fait ~150 jetons)
}
KEEP_ALIVE = "30m"       # garde le modèle en mémoire entre deux items (et deux lancements proches)
 
HERE = Path(__file__).resolve().parent          # .../src/questions
ROOT = HERE.parents[1]                          # racine du dépôt
PROMPT_FILE = ROOT / "configs" / "systems_prompt" / "system_prompt.txt"
ITEMS_FILE = HERE / "items.json"
OUT_DIR = ROOT / "res" / "questions"
OUT_JSON = OUT_DIR / "concepts_questions.json"
OUT_CSV = OUT_DIR / "concepts_questions.csv"
CACHE_FILE = OUT_DIR / "raw_outputs_cache.json"  # réponses brutes de Qwen, item par item
 
CSV_FIELDS = ["item_id", "concept_proposed", "category", "relation", "evidence", "synonyms"]
 
 
def load_items() -> list[dict]:
    """Aplatit items.json en une liste d'items (le questionnaire et l'échelle sont
    repris du bloc du questionnaire)."""
    data = load_json_file(str(ITEMS_FILE))
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
    return (
        f"Questionnaire: {item['questionnaire']}\n"
        f"Item ID: {item['item_id']}\n"
        f'Item text: "{item["question"]}"\n'
        f"Response type: {item['response_type']}\n"
        f"Response scale: {item['scale']}"
    )
 
 
def ask_qwen(system_prompt: str, item: dict, user_prompt: str):
    """Un appel à Qwen pour un item. Renvoie le JSON du modèle (dict) ou None."""
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
    # L'identifiant vient du fichier d'items, jamais du modèle
    data["item_id"] = item["item_id"]
    return data
 
 
def make_signature(system_prompt: str) -> str:
    """Empreinte du prompt, du modèle et des options : si l'un change, le cache est ignoré."""
    text = system_prompt + MODEL + json.dumps(OPTIONS, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
 
 
def load_cache(signature: str) -> dict:
    if CACHE_FILE.exists():
        try:
            data = load_json_file(str(CACHE_FILE))
        except json.JSONDecodeError:
            return {}
        if data.get("signature") == signature:
            return data.get("items", {})
        print("Cache ignoré : le prompt, le modèle ou les options ont changé.")
    return {}
 
 
def save_cache(signature: str, items: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_file(str(CACHE_FILE), {"signature": signature, "items": items})
 
 
def format_relation(relation) -> str:
    if not relation or not isinstance(relation, dict):
        return ""
    return f"{relation.get('predicate', '')} -> {relation.get('target_concept', '')}"
 
 
def main() -> None:
    system_prompt = load_txt_file(str(PROMPT_FILE))
    items = load_items()
 
    signature = make_signature(system_prompt)
    cache = load_cache(signature)
    start = time.time()
 
    raw_outputs = []
    for i, item in enumerate(items, 1):
        iid = item["item_id"]
        user_prompt = build_user_prompt(item)
        entry = cache.get(iid)
        if entry and entry.get("user_prompt") == user_prompt:
            raw = entry["raw"]
            print(f"[{i}/{len(items)}] {iid} : repris du cache (aucun appel à Qwen)")
        else:
            t0 = time.time()
            raw = ask_qwen(system_prompt, item, user_prompt)
            print(f"[{i}/{len(items)}] {iid} : {time.time() - t0:.0f} s")
            if raw is None:
                print(f"  ATTENTION {iid} : sortie JSON invalide du modèle")
            else:
                cache[iid] = {"user_prompt": user_prompt, "raw": raw}
                save_cache(signature, cache)  # sauvegarde après chaque item
        raw_outputs.append(raw)
 
    # UN GROUPE PAR ITEM : postprocess supprime les doublons à l'intérieur d'un groupe.
    # Si tous les items étaient dans le même groupe, deux items au même concept
    # (epw4 / epw8) fusionneraient et l'un disparaîtrait.
    cleaned = postprocess([[raw] for raw in raw_outputs])
 
    results, missing = [], []
    for item, concepts in zip(items, cleaned):
        if not concepts:
            missing.append(item["item_id"])
            print(f"  ATTENTION {item['item_id']} : aucun concept conservé par postprocess")
        results.extend(c.to_dict() for c in concepts)
 
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_file(str(OUT_JSON), results)
 
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "item_id": r["item_id"],
                "concept_proposed": r["concept_proposed"],
                "category": r["category"],
                "relation": format_relation(r["relation"]),
                "evidence": r["evidence"],
                "synonyms": "; ".join(r["synonyms"]),
            })
 
    print(f"\nTerminé en {(time.time() - start) / 60:.1f} min : "
          f"{len(items)} items, {len(results)} concept(s) écrits.")
    if missing:
        print(f"Items sans concept conservé : {', '.join(missing)}")
    print(f"JSON : {OUT_JSON}\nCSV  : {OUT_CSV}")
 
 
if __name__ == "__main__":
    main()
 