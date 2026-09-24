import hashlib
import json
import re
import time
from datetime import date
from pathlib import Path
import ollama

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# 1. Les 10 variables sélectionnées 
VARIABLES = [
    {"name": "RheumatoidArthritis", "type": "binary numeric", "example": "0 / 1",
     "description": "Rheumatoid arthritis is a type of arthritis where your immune system attacks the tissue lining the joints on both sides of your body. It may affect other parts of your body, too. The exact cause is unknown. Treatment options include lifestyle changes, physical therapy, occupational therapy, nutritional therapy, medication and surgery."},
    {"name": "BRI", "type": "continuous numeric", "example": "4.82",
     "description": " body roundness index (BRI) – may be more effective at measuring a person’s health. Whereas BMI is determined using one’s height and weight only, BRI also includes an individual’s waist circumference (or roundness) in the calculation. The result is a clearer picture of how body fat is distributed: The more fat in the middle of your body, the more at-risk you are for some health conditions., unitless"},
    {"name": "Age", "type": "integer", "example": "22",
     "description": "age in years"},
    {"name": "SmokingStatus", "type": "categorical text", "example": "Former",
     "description": "a standard classification that describes a person's current and past habits regarding tobacco use"},
    {"name": "BMI", "type": "continuous numeric", "example": "27.4",
     "description": "Body mass index (BMI) is a tool that uses your height and weight to estimate your body fat. It can help identify your risk for certain health conditions, like heart disease. You can calculate your BMI using a simple math formula. BMI has limitations because it doesn’t consider all aspects of your health. in kg/m2"},
    {"name": "DrinkingStatus", "type": "ordinal categorical text", "example": "Occasional drinker",
     "description": "is a standardized clinical term used to document a patient's current and historical patterns of alcohol consumption."},
    {"name": "Hypertension", "type": "binary categorical text", "example": "Normal",
     "description": " Blood pressure is the force of blood pushing against the walls of your arteries as your heart pumps. HTN means this pressure is consistently too high."},
    {"name": "Diabetes", "type": "categorical text", "example": "Prediabetes",
     "description": "a chronic condition that causes your blood sugar (glucose) levels to become too high"},
    {"name": "Hyperlipidemia", "type": "binary categorical text", "example": "Normal",
     "description": "Hyperlipidemia (high cholesterol) means you have too many lipids (fats) in your blood, which increases your risk of heart attack and stroke."},
    {"name": "PhysicalActivity", "type": "ordinal categorical text", "example": "Moderate activity",
     "description": "any bodily movement produced by skeletal muscles that requires energy expenditure"},
]

PROMPT_TEMPLATE = """You are a medical ontology assistant tasked with organizing the {n} variables below.

For each variable, propose:
- a normalized concept,
- a semantic category (Symptom, Disease, Treatment, Biomarker, Assessment, Demographic, Lifestyle, etc.),
- one relevant relation, when appropriate.

Use only the information provided.
Return one simple result per variable.

Answer strictly as a JSON array, one object per variable, with the keys:
"variable", "normalized_concept", "semantic_category", "relation".

Variables:
{variables}
"""


def build_prompt(variables=VARIABLES) -> str:
    """Insère la liste de variables dans PROMPT_TEMPLATE pour former le prompt final envoyé à Qwen."""
    lines = [
        f"- {v['name']} | type: {v['type']} | example: {v['example']} | description: {v['description']}"
        for v in variables
    ]
    return PROMPT_TEMPLATE.format(n=len(variables), variables="\n".join(lines))


def ask_qwen(prompt: str, model: str = "qwen2.5:7b") -> str:
    """Envoie le prompt à Qwen via Ollama (en local) et renvoie le texte brut de sa réponse."""
    print(f"-> Envoi du prompt au modèle local '{model}' (Ollama)...")
    start = time.time()
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},  # réponses reproductibles
    )
    print(f"-> Réponse reçue en {time.time() - start:.1f}s.")
    return response["message"]["content"]


def parse_json(raw: str) -> list[dict]:
    """Extrait le tableau JSON même s'il est entouré de texte ou de balises Markdown."""
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise ValueError("Aucun tableau JSON trouvé dans la sortie du modèle.")
    return json.loads(match.group(0))


def prompt_id(prompt: str) -> str:
    """ID court et stable dérivé du contenu du prompt (même prompt -> même ID)."""
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]


def save(prompt: str, rows: list[dict], variables=VARIABLES) -> Path:
    """Complète chaque ligne proposée par Qwen avec 'evidence' (généré) et les champs
    de validation manuelle vides, puis écrit le tout dans outputs/qwen_resultats_<date>_<id>.json."""
    metadata = {v["name"]: v for v in variables}
    for row in rows:
        var = metadata.get(row["variable"], {})
        row["evidence"] = f"type: {var.get('type', '?')} | example: {var.get('example', '?')} | description: {var.get('description', '?')}"
        # à remplir après recherche manuelle dans SNOMED CT ou une autre ontologie biomédicale
        row.setdefault("source_ontology", "")  # ex: "SNOMED CT", "LOINC", "ICD-10"
        row.setdefault("preferred_label", "")
        row.setdefault("identifier_uri", "")
        row.setdefault("match", "")  # Exact / Broader / Narrower / No Match
        row.setdefault("status", "")  # Existing concept / Candidate Project Concept (si No Match)
        row.setdefault("validation_status", "Uncertain")  # Correct / Incorrect / Uncertain

    stamp = date.today().isoformat()
    pid = prompt_id(prompt)
    output = {"prompt": prompt, "results": rows}
    out_path = OUT_DIR / f"qwen_resultats_{stamp}_{pid}.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    prompt = build_prompt()
    print(prompt)
    raw = ask_qwen(prompt)
    rows = parse_json(raw)
    out_path = save(prompt, rows)
    print(f"{len(rows)} variables traitées -> '{out_path}'")