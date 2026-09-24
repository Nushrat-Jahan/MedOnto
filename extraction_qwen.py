"""
Extraction de concepts cliniques à partir d'items de questionnaire (PHQ-9, GAD-7, ESS)
via Qwen2.5-7B-Instruct (Ollama en local).
 
Prérequis : pip install ollama --break-system-packages
            ollama doit tourner en arrière-plan (le modèle qwen2.5:7b-instruct déjà téléchargé)
 
Sortie : un fichier CSV avec, pour chaque item, le concept/catégorie/relation
proposés par Qwen. Les colonnes SNOMED CT match / Status / Validation restent
à compléter manuellement après vérification sur browser.ihtsdotools.org.
"""
 
import json
import csv
import ollama
 
MODEL = "qwen2.5:7b-instruct"
 
# --- Étape 2 : les items sélectionnés (à adapter si tu changes ta sélection) ---
ITEMS = [
    {"questionnaire": "PHQ-9", "item_id": "phq1",
     "question": "Little interest or pleasure in doing things",
     "response_type": "Frequency over past 2 weeks",
     "scale": "0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day"},
    {"questionnaire": "PHQ-9", "item_id": "phq3",
     "question": "Trouble falling or staying asleep, or sleeping too much",
     "response_type": "Frequency over past 2 weeks",
     "scale": "0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day"},
    {"questionnaire": "PHQ-9", "item_id": "phq9",
     "question": "Thoughts that you would be better off dead, or of hurting yourself in some way",
     "response_type": "Frequency over past 2 weeks",
     "scale": "0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day"},
    {"questionnaire": "GAD-7", "item_id": "gad1",
     "question": "Feeling nervous, anxious, or on edge",
     "response_type": "Frequency over past 2 weeks",
     "scale": "0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day"},
    {"questionnaire": "GAD-7", "item_id": "gad3",
     "question": "Worrying too much about different things",
     "response_type": "Frequency over past 2 weeks",
     "scale": "0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day"},
    {"questionnaire": "GAD-7", "item_id": "gad7",
     "question": "Feeling afraid as if something awful might happen",
     "response_type": "Frequency over past 2 weeks",
     "scale": "0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day"},
    {"questionnaire": "Epworth Sleepiness Scale", "item_id": "epw1",
     "question": "Chance of dozing while sitting and reading",
     "response_type": "Likelihood of dozing",
     "scale": "0=Would never doze, 1=Slight chance, 2=Moderate chance, 3=High chance"},
    {"questionnaire": "Epworth Sleepiness Scale", "item_id": "epw4",
     "question": "Chance of dozing as a passenger in a car for an hour without a break",
     "response_type": "Likelihood of dozing",
     "scale": "0=Would never doze, 1=Slight chance, 2=Moderate chance, 3=High chance"},
]
 
# --- Étape 3 : prompt d'extraction ---
SYSTEM_PROMPT = """You are a clinical terminology expert helping build an ontology.
For each questionnaire item you receive, respond ONLY with a JSON object (no preamble,
no markdown fences) with exactly these fields:
{
  "concept": "the normalized clinical concept measured by this item (short noun phrase)",
  "category": "one of: symptom, disease, treatment, biomarker, assessment",
  "relation": "one short relation verb linking the item to the concept, e.g. 'measures' or 'indicates'",
  "evidence": "one short sentence explaining why, based on the question wording"
}"""
 
 
def build_user_prompt(item: dict) -> str:
    return (
        f"Questionnaire: {item['questionnaire']}\n"
        f"Item ID: {item['item_id']}\n"
        f"Question: {item['question']}\n"
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
 
 
def main():
    rows = []
    for item in ITEMS:
        print(f"→ Extraction pour {item['item_id']} ...")
        try:
            result = extract_concept(item)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Erreur sur {item['item_id']} : {e}")
            result = {"concept": "", "category": "", "relation": "", "evidence": ""}
 
        rows.append({
            "Source": f"{item['questionnaire']} - {item['item_id']} : {item['question']}",
            "Concept": result.get("concept", ""),
            "Category": result.get("category", ""),
            "Relation": result.get("relation", ""),
            "Evidence": result.get("evidence", ""),
            "SNOMED CT match": "",   # à compléter manuellement (browser.ihtsdotools.org)
            "Status": "",            # Existing concept / candidate project concept
            "Validation": "",        # Correct / Incorrect / Uncertain
        })
 
    fieldnames = ["Source", "Concept", "Category", "Relation", "Evidence",
                  "SNOMED CT match", "Status", "Validation"]
    with open("resultats_extraction.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
 
    print("\nTerminé. Résultats écrits dans resultats_extraction.csv")
 
if __name__ == "__main__":
    main()
 