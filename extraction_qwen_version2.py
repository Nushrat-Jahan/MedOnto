"""
Extraction de concepts cliniques à partir d'items de questionnaire (PHQ-9, GAD-7, ESS)
via Qwen2.5-7B-Instruct (Ollama en local).  -- VERSION 3

Prérequis : pip install ollama --break-system-packages
            ollama doit tourner en arrière-plan (le modèle qwen2.5:7b-instruct déjà téléchargé)

Sortie : resultats_extraction_v3.csv et resultats_extraction_v3.json.
Les colonnes SNOMED CT match / Validation restent à compléter manuellement après
vérification sur browser.ihtsdotools.org (Qwen n'a PAS accès à SNOMED CT et ne doit
pas inventer de codes). Le Status proposé par Qwen est provisoire.
"""

import json
import csv
import ollama

MODEL = "qwen2.5:7b-instruct"

# Échelles (identiques pour tous les items d'un même questionnaire)
SCALE_FREQ = "0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day"
SCALE_ESS = "0=Would never doze, 1=Slight chance of dozing, 2=Moderate chance of dozing, 3=High chance of dozing"
TYPE_FREQ = "Likert 4 points - frequency over the past 2 weeks"
TYPE_ESS = ("Likert 4 points - how likely the person is to doze off or fall asleep in this "
            "situation (sleepiness, as opposed to feeling merely tired)")

# --- Étape 2 : les 9 items sélectionnés (formulations officielles) ---
ITEMS = [
    {"questionnaire": "PHQ-9", "item_id": "phq2",
     "question": "Feeling down, depressed, or hopeless",
     "response_type": TYPE_FREQ, "scale": SCALE_FREQ},
    {"questionnaire": "PHQ-9", "item_id": "phq4",
     "question": "Feeling tired or having little energy",
     "response_type": TYPE_FREQ, "scale": SCALE_FREQ},
    {"questionnaire": "PHQ-9", "item_id": "phq7",
     "question": "Trouble concentrating on things, such as reading the newspaper or watching television",
     "response_type": TYPE_FREQ, "scale": SCALE_FREQ},
    {"questionnaire": "PHQ-9", "item_id": "phq9",
     "question": "Thoughts that you would be better off dead or of hurting yourself in some way",
     "response_type": TYPE_FREQ, "scale": SCALE_FREQ},
    {"questionnaire": "GAD-7", "item_id": "gad1",
     "question": "Feeling nervous, anxious, or on edge",
     "response_type": TYPE_FREQ, "scale": SCALE_FREQ},
    {"questionnaire": "GAD-7", "item_id": "gad4",
     "question": "Trouble relaxing",
     "response_type": TYPE_FREQ, "scale": SCALE_FREQ},
    {"questionnaire": "GAD-7", "item_id": "gad6",
     "question": "Becoming easily annoyed or irritable",
     "response_type": TYPE_FREQ, "scale": SCALE_FREQ},
    {"questionnaire": "Epworth Sleepiness Scale", "item_id": "epw4",
     "question": "As a passenger in a car for an hour without a break",
     "response_type": TYPE_ESS, "scale": SCALE_ESS},
    {"questionnaire": "Epworth Sleepiness Scale", "item_id": "epw8",
     "question": "In a car, while stopped for a few minutes in traffic",
     "response_type": TYPE_ESS, "scale": SCALE_ESS},
]

CATEGORIES = {"finding", "symptom", "disorder", "behavior", "observable_entity", "assessment_scale"}
PREDICATES = {"is_a", "has_associated_finding", "has_interpretation", "measures", "is_manifestation_of"}
STATUSES = {"existing_concept", "candidate_project_concept"}

# --- Étape 3 : prompt d'extraction (v3) ---
SYSTEM_PROMPT = """You are a clinical terminology assistant specialized in mapping psychometric
questionnaire items to standardized biomedical concepts (SNOMED CT semantics).
You do not have live access to a SNOMED CT database. You must NOT invent SNOMED CT codes.
Instead, propose the clinical concept in precise medical English, as it would appear as a
SNOMED CT preferred term, so that a human annotator can verify it in the terminology browser.

Rules:
1. Distinguish the underlying clinical construct (e.g. "anhedonia") from the item's surface
   wording (e.g. "little interest or pleasure in doing things").
2. Each item is a self-report screening question. Map it to the specific symptom or clinical
   finding it asks about, not to the disorder that the questionnaire screens for.
3. Never answer with generic labels such as "symptom", "mental health issue" or
   "psychological state". Always name the specific clinical finding.
4. "category" must be exactly one of:
   finding, symptom, disorder, behavior, observable_entity, assessment_scale.
5. "relation" is optional: propose one only if it is clearly supported by the item wording,
   otherwise use null. Use the form {"predicate": "...", "target_concept": "..."} with
   "predicate" chosen from: is_a, has_associated_finding, has_interpretation, measures,
   is_manifestation_of. The target_concept must be different from concept_proposed.
6. "evidence" must be ONE short sentence (25 words maximum) in your own words explaining why
   the concept fits this item. Do not copy or restate the item text or the response scale, and
   do not mention any disorder or diagnosis that is not in the item.
7. If you are not confident that a standard clinical concept exists, set "status" to
   "candidate_project_concept" instead of guessing a false match.
8. For Epworth Sleepiness Scale items, the item text is only a situation used to probe general
   daytime sleepiness. The concept measured is the sleepiness itself, not the situation
   (car, TV, reading, driving) and not a diagnosis.
9. Output strictly valid JSON, with no preamble and no markdown fences.

Return exactly this JSON structure:
{
  "item_id": "...",
  "concept_proposed": "...",
  "category": "...",
  "relation": {"predicate": "...", "target_concept": "..."} or null,
  "evidence": "...",
  "status": "existing_concept" or "candidate_project_concept"
}"""


def build_user_prompt(item: dict) -> str:
    return (
        f"Questionnaire: {item['questionnaire']}\n"
        f"Item ID: {item['item_id']}\n"
        f"Item text: \"{item['question']}\"\n"
        f"Response type: {item['response_type']}\n"
        f"Response scale: {item['scale']}"
    )


def extract_concept(item: dict) -> dict:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(item)},
        ],
        options={"temperature": 0},  # sortie stable et reproductible
        format="json",               # force une sortie JSON valide
    )
    raw = response["message"]["content"]
    return json.loads(raw)


def format_relation(relation) -> str:
    """Aplatit l'objet relation en 'predicate -> target' pour la colonne CSV."""
    if not relation or not isinstance(relation, dict):
        return ""
    return f"{relation.get('predicate', '')} -> {relation.get('target_concept', '')}"


def check_result(item: dict, result: dict) -> None:
    """Signale les sorties qui sortent du schéma demandé (à revérifier à la main)."""
    item_id = item["item_id"]
    if result.get("category") not in CATEGORIES:
        print(f"  ATTENTION {item_id} : catégorie hors liste -> {result.get('category')!r}")
    rel = result.get("relation")
    if isinstance(rel, dict):
        if rel.get("predicate") not in PREDICATES:
            print(f"  ATTENTION {item_id} : prédicat hors liste -> {rel.get('predicate')!r}")
        target = str(rel.get("target_concept", "")).strip().lower()
        if target and target == str(result.get("concept_proposed", "")).strip().lower():
            print(f"  ATTENTION {item_id} : relation tautologique (cible = concept)")
    if result.get("status") not in STATUSES:
        print(f"  ATTENTION {item_id} : status hors liste -> {result.get('status')!r}")
    evidence = str(result.get("evidence", "")).strip().lower()
    question = item["question"].strip().lower()
    if not evidence:
        print(f"  ATTENTION {item_id} : evidence vide")
    elif evidence in question or question in evidence:
        print(f"  ATTENTION {item_id} : evidence recopie l'item")


def main():
    rows = []
    raw_results = []  # on garde aussi le JSON brut, c'est la sortie "JSON" attendue
    for item in ITEMS:
        print(f"→ Extraction pour {item['item_id']} ...")
        try:
            result = extract_concept(item)
            check_result(item, result)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Erreur sur {item['item_id']} : {e}")
            result = {"concept_proposed": "", "category": "", "relation": None,
                      "evidence": "", "status": ""}

        raw_results.append({"item_id": item["item_id"], **result})
        rows.append({
            "Source": f"{item['questionnaire']} - {item['item_id']} : {item['question']}",
            "Concept": result.get("concept_proposed", ""),
            "Category": result.get("category", ""),
            "Relation": format_relation(result.get("relation")),
            "Evidence": result.get("evidence", ""),
            "SNOMED CT match": "",   # à compléter manuellement (browser.ihtsdotools.org)
            "Status": result.get("status", ""),  # provisoire, à confirmer après vérif SNOMED
            "Validation": "",        # Correct / Incorrect / Uncertain
        })

    fieldnames = ["Source", "Concept", "Category", "Relation", "Evidence",
                  "SNOMED CT match", "Status", "Validation"]
    with open("resultats_extraction_v3.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open("resultats_extraction_v3.json", "w", encoding="utf-8") as f:
        json.dump(raw_results, f, ensure_ascii=False, indent=2)

    print("\nTerminé. Résultats écrits dans resultats_extraction_v3.csv et resultats_extraction_v3.json")


if __name__ == "__main__":
    main()