"""
Étape C2 du pipeline automatique — Décision du match par Qwen.

Le modèle reçoit, pour chaque item, le texte de l'item ET la liste FERMÉE
des candidats SNOMED CT produits par B2 (label + ID + définition + synonymes),
et choisit le meilleur candidat ou "No Match", classé :
    Exact / Broader / Narrower / No Match

Contrainte stricte anti-hallucination, appliquée par le SCRIPT :
  - le candidate_id renvoyé doit appartenir à la liste B2 de l'item,
    sinon le résultat est forcé à "No Match" (et logué) ;
  - le label retenu est repris de B2 (l'ID est la source de vérité,
    jamais la sortie du modèle) ;
  - un item sans candidat B2 est déclaré "No Match" sans appel au LLM ;
  - match hors de l'enum -> "No Match".

C2 ne fait que choisir/juger : il ne produit jamais de valeur SNOMED
absente des réponses de l'API BioPortal.

Sortie : resultats_C2_match.csv
"""

import csv
import json
from concurrent.futures import ThreadPoolExecutor

import ollama

MODEL = "qwen2.5:7b-instruct"
MAX_WORKERS = 1
VALID_MATCHES = {"Exact", "Broader", "Narrower", "No Match"}
INPUT_A2 = "resultats_A2.json"
INPUT_B2 = "resultats_B2_candidats.csv"
OUTPUT_CSV = "resultats_C2_match.csv"

SYSTEM_PROMPT = """You are a clinical terminology matching judge.
You receive (1) a questionnaire item and (2) a CLOSED list of real SNOMED CT
candidates already retrieved and verified by the BioPortal API (id, preferred
label, definition).
Your only task: choose the single candidate that best corresponds to the
clinical construct the item measures, and classify the match.
Do NOT search, do NOT recall other SNOMED concepts, do NOT invent anything.

Match values (scope of the candidate relative to the item's construct):
- "Exact"    : candidate and construct are equivalent in scope.
- "Broader"  : candidate addresses the SAME clinical construct but is more general
               (the item is a specific case of it).
- "Narrower" : candidate addresses the SAME clinical construct but is more specific
               (the item covers more than the candidate alone).
- "No Match" : NO candidate addresses the same clinical construct at all (wrong
               clinical domain, e.g. a sleep concept for an anxiety item).

Decision priority: first check whether ANY candidate targets the same underlying
construct as the item, even partially. If yes, you MUST classify it as Exact,
Broader, or Narrower — a partial or imperfect scope match is NEVER a reason to
answer "No Match". Reserve "No Match" strictly for cases where every candidate
belongs to a different clinical domain or concept altogether.

Important: "construct" means the clinical/symptomatic content only. It does NOT
include the response scale's timeframe or frequency wording (e.g. "over the past
2 weeks", "several days") — these describe how often the symptom occurs, not what
the symptom is. For Epworth Sleepiness Scale items, the situational context (car,
TV, reading...) is only a probe used to test general daytime sleepiness; match
against the sleepiness concept itself, never against the situation. Do not reject
a candidate merely because it does not restate the timeframe, frequency, or
situational wrapper of the item.

Hard rules:
1. You may only reference a candidate by its exact "id" as listed. Never invent,
   modify, guess or complete any SNOMED id or label that is not in the list.
2. If no candidate truly fits, answer "No Match" with candidate_id = null.
3. Judge the construct measured by the item, not the surface wording.
4. "justification" = one short sentence (max 25 words).
5. Output strictly valid JSON, no preamble, no markdown fences.

Return exactly:
{"item_id": "...", "match": "Exact|Broader|Narrower|No Match",
 "candidate_id": "<id from the list>" or null,
 "justification": "..."}"""


def build_user_prompt(item: dict, candidates: list) -> str:
    lines = [
        f"Questionnaire: {item['questionnaire']}",
        f"Item ID: {item['item_id']}",
        f'Item text: "{item["question"]}"',
        f"Response type: {item['response_type']}",
        f"Response scale: {item['scale']}",
        "",
        "Closed list of real SNOMED CT candidates (from the BioPortal API):",
    ]
    for i, c in enumerate(candidates, 1):
        lines.append(
            f"{i}. id={c['SNOMED ID']} | label={c['Preferred label']} "
            f"| definition={c['Definition']} | synonyms={c['Synonyms']}"
        )
    return "\n".join(lines)


def make_row(item: dict, match: str, cid: str, label: str, why: str) -> dict:
    return {
        "Item ID": item["item_id"],
        "Item text": item["question"],
        "Concept (A2)": item.get("concept_proposed", ""),
        "Match": match,
        "SNOMED ID": cid,
        "Preferred label": label,
        "Justification": why,
    }


def decide(item: dict, candidates: list) -> dict:
    ids = {c["SNOMED ID"] for c in candidates}
    label_by_id = {c["SNOMED ID"]: c["Preferred label"] for c in candidates}

    if not candidates:
        return make_row(item, "No Match", "", "",
                        "Aucun candidat renvoyé par BioPortal (B2) pour ce concept.")

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(item, candidates)},
        ],
        options={"temperature": 0},
        format="json",
    )
    try:
        raw = json.loads(response["message"]["content"])
    except json.JSONDecodeError:
        print(f"  ATTENTION {item['item_id']} : sortie JSON invalide -> No Match")
        return make_row(item, "No Match", "", "", "Sortie JSON invalide du modèle.")

    match = str(raw.get("match", "")).strip()
    cid = str(raw.get("candidate_id") or "").strip()
    justification = str(raw.get("justification", "")).strip()

    if match not in VALID_MATCHES:
        print(f"  REJET {item['item_id']} : match {match!r} hors de l'enum -> No Match")
        return make_row(item, "No Match", "", "",
                        f"Sortie hors enum ({match!r}) ; No Match forcé par le script.")

    if match == "No Match":
        return make_row(item, "No Match", "", "", justification)

    if cid not in ids:
        # Garde-fou anti-hallucination : ID absent de la liste B2 -> No Match
        print(f"  REJET anti-hallucination {item['item_id']} : "
              f"id {cid!r} absent de la liste B2 -> No Match")
        return make_row(item, "No Match", "", "",
                        f"Candidat {cid!r} absent de la liste B2 ; No Match forcé par le script.")

    # Le label est repris de B2 (source de vérité), jamais de la sortie du LLM
    return make_row(item, match, cid, label_by_id[cid], justification)


def main() -> None:
    with open(INPUT_A2, encoding="utf-8") as f:
        items = json.load(f)
    items_by_id = {it["item_id"]: it for it in items}

    cands_by_item = {}
    with open(INPUT_B2, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cands_by_item.setdefault(row["Item ID"].strip(), []).append(row)

    def work(iid: str) -> dict:
        item = items_by_id[iid]
        cands = cands_by_item.get(iid, [])
        print(f"→ Décision pour {iid} ({len(cands)} candidat(s) B2) ...")
        return decide(item, cands)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        rows = list(pool.map(work, [it["item_id"] for it in items]))

    fieldnames = ["Item ID", "Item text", "Concept (A2)", "Match",
                  "SNOMED ID", "Preferred label", "Justification"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nTerminé. Décisions écrites dans {OUTPUT_CSV}")


if __name__ == "__main__":
    main()