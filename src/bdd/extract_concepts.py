"""Étape 2 du pipeline : extraction des concepts par Qwen, à partir des variables extraites.

Pour chaque variable de variables.json (étape 1), le script construit un prompt, l'envoie à
Qwen (en local via Ollama) et récupère sa proposition de concept au format commun du projet :
{"item_id", "concept_proposed", "category", "relation", "evidence", "status"}.
Les variables sont envoyées une par une : un prompt court et une réponse simple, plus fiables
pour un modèle de 7 milliards de paramètres.

Test rapide, depuis la racine du projet :
    python src/bdd/extract_concepts.py "outputs/<nom_base>_<id>/variables.json"
"""

import hashlib
import json
import sys
from pathlib import Path
import ollama
import pandas as pd

MODEL = "qwen2.5:7b"  # modèle installé localement dans Ollama (le même qu'en semaine 1)

# Valeurs autorisées dans la réponse de Qwen (mêmes listes que dans SYSTEM_PROMPT)
CATEGORIES = ["procedure", "observable_entity", "finding", "disorder", "substance", "qualifier_value"]
STATUSES = ["existing_concept", "candidate_project_concept"]

SYSTEM_PROMPT = """You are a clinical terminology assistant. You map one variable of a structured
health dataset (one column of a table) to a standardized biomedical concept (SNOMED CT semantics).
You do not have live access to SNOMED CT. You must NOT invent SNOMED CT codes.
Instead, propose the concept in precise medical English, as it would appear as a SNOMED CT
preferred term, so that it can be searched in SNOMED CT afterwards.

Rules:
1. Use everything you are given: the variable name, its label, unit, type and values, and the
   dataset documentation. The label and the values may be written in another language
   (e.g. French): the proposed concept must always be in English.
2. Name what the column records, not a disease it may be related to. For example, a column of
   blood glucose values in mmol/L records a "glucose measurement", not "diabetes mellitus".
3. For a yes/no variable, name what "yes" stands for.
4. "category" must be exactly one of these SNOMED CT hierarchies:
   procedure, observable_entity, finding, disorder, substance, qualifier_value.
   A laboratory test or measurement is a procedure.
5. "relation" is optional: propose one only if it is clearly supported by the variable,
   otherwise use null. Use the form {"predicate": "...", "target_concept": "..."} with
   "predicate" chosen from: is_a, has_associated_finding, has_interpretation, measures,
   is_manifestation_of. The target_concept must be different from concept_proposed.
6. "evidence" must be ONE short sentence (25 words maximum) explaining why the concept fits
   the variable.
7. If you are not confident that a standard SNOMED CT concept exists, set "status" to
   "candidate_project_concept" instead of guessing.
8. Output strictly valid JSON, with no preamble and no markdown fences.

Return exactly this JSON structure:
{
  "item_id": "<the variable name>",
  "concept_proposed": "...",
  "category": "...",
  "relation": {"predicate": "...", "target_concept": "..."} or null,
  "evidence": "...",
  "status": "existing_concept" or "candidate_project_concept"
}"""


def build_user_prompt(variable: dict, readme: list[dict] | None = None) -> str:
    """Rédige le message envoyé à Qwen pour une variable, à partir de sa fiche (étape 1).

    Le message contient :
    - la documentation de la base (README), si elle existe, recopiée ligne par ligne ;
    - la description de la variable : nom, libellé, unité, type, valeurs possibles,
      intervalle et exemple. Seuls les champs renseignés sont écrits : une variable sans
      libellé n'a pas de ligne "Label".

    Exemple pour la variable BIO_CRP d'un modèle BIO :
        Variable name: BIO_CRP
        Label: Bio. : CRP (mg/L)
        Unit: mg/L
        Type: numeric

    Paramètres :
        variable : une fiche de la liste renvoyée par extract_variables.
        readme   : la documentation renvoyée par load_readme, ou None.

    Renvoie :
        le texte du message, en anglais (sauf les libellés et valeurs, recopiés tels quels).
    """
    lines = []

    # Documentation de la base, ex. "- Codes: CSV and Excel use Oui/Non ..."
    if readme is not None:
        lines.append("Dataset documentation:")
        for row in readme:
            texts = []
            for value in row.values():
                if not pd.isna(value):  # cellule vide du README
                    texts.append(str(value))
            lines.append("- " + ": ".join(texts))
        lines.append("")

    # Description de la variable : seulement les champs renseignés
    lines.append(f"Variable name: {variable['name']}")
    if variable.get("label") is not None:
        lines.append(f"Label: {variable['label']}")
    if variable.get("unit") is not None:
        lines.append(f"Unit: {variable['unit']}")
    lines.append(f"Type: {variable['type']}")
    if variable.get("values") is not None:
        values = [str(value) for value in variable["values"]]
        lines.append(f"Possible values: {' / '.join(values)}")
    if variable.get("range") is not None:
        lines.append(f"Range: {variable['range']}")
    if variable.get("example") is not None:
        lines.append(f"Example: {variable['example']}")

    return "\n".join(lines)


def ask_qwen(user_prompt: str) -> str:
    """Envoie le prompt système et le message d'une variable à Qwen, et renvoie sa réponse brute.

    Qwen tourne en local via Ollama (le logiciel Ollama doit être lancé). Deux réglages :
    - temperature=0 : Qwen choisit toujours le mot le plus probable, donc le même prompt
      donne toujours la même réponse (résultats reproductibles) ;
    - format="json" : Ollama oblige Qwen à répondre en JSON valide.

    Paramètre :
        user_prompt : le message rédigé par build_user_prompt.

    Renvoie :
        le texte de la réponse de Qwen, tel quel.
    """
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0},
        format="json",
    )
    return response["message"]["content"]


def parse_concept(raw: str) -> dict | None:
    """Transforme la réponse brute de Qwen en dictionnaire Python.

    Paramètre :
        raw : la réponse renvoyée par ask_qwen.

    Renvoie :
        le dictionnaire du concept, ex. {"item_id": "BIO_CRP", "concept_proposed": ..., ...} ;
        None si la réponse n'est pas un objet JSON valide.
    """
    try:
        concept = json.loads(raw)
    except json.JSONDecodeError:
        return None

    # On attend un seul objet {...}, pas une liste ni un nombre
    if not isinstance(concept, dict):
        return None
    return concept


def extract_concepts(variables: list[dict], readme: list[dict] | None = None) -> list[dict]:
    """Demande à Qwen un concept pour chaque variable de la base, une variable à la fois.

    Les identifiants (ex. numéro de patient) sont ignorés : ce ne sont pas des variables
    médicales. Les réponses de Qwen ne sont jamais corrigées par le script. Celles qui sortent
    du format demandé (JSON invalide, catégorie ou statut hors liste) sont signalées à l'écran,
    pour être revérifiées à la main.

    Pour chaque variable, on garde tout ce qui permet de refaire et de vérifier l'expérience :
    le message envoyé, la réponse brute de Qwen et le concept lu dans cette réponse.

    Paramètres :
        variables : la liste renvoyée par extract_variables (ou lue dans variables.json).
        readme    : la documentation renvoyée par load_readme, ou None.

    Renvoie :
        une liste avec un dictionnaire par variable envoyée à Qwen :
        {"variable", "user_prompt", "raw_output", "concept"}
        ("concept" vaut None si la réponse n'est pas un JSON valide).
    """
    concepts = []

    for variable in variables:
        # Un identifiant n'est pas une variable médicale : il n'est pas envoyé à Qwen
        if variable["type"] == "identifier":
            continue

        print(f"-> Qwen : {variable['name']}...")
        user_prompt = build_user_prompt(variable, readme)
        raw = ask_qwen(user_prompt)
        concept = parse_concept(raw)

        # Signalement des réponses à revérifier (rien n'est corrigé)
        if concept is None:
            print("   ATTENTION : la réponse n'est pas un JSON valide")
        else:
            if concept.get("category") not in CATEGORIES:
                print(f"   ATTENTION : catégorie hors liste -> {concept.get('category')}")
            if concept.get("status") not in STATUSES:
                print(f"   ATTENTION : statut hors liste -> {concept.get('status')}")

        concepts.append({
            "variable": variable["name"],
            "user_prompt": user_prompt,
            "raw_output": raw,
            "concept": concept,
        })

    return concepts


def save_concepts(concepts: list[dict], dataset_dir: str | Path) -> Path:
    """Enregistre les concepts proposés par Qwen dans le dossier de la base.

    Le fichier s'appelle concepts_<id du prompt>.json, où l'ID est l'empreinte SHA-1 du
    prompt système, réduite à 8 caractères (même principe que dataset_id) :
    - même prompt -> même fichier : comme temperature=0, la réponse est la même ;
    - prompt modifié -> nouveau fichier : les résultats des anciens prompts sont conservés,
      ce qui permet de comparer les versions du prompt.

    Le fichier contient en en-tête le modèle et le prompt système exact, puis les concepts.

    Paramètres :
        concepts    : la liste renvoyée par extract_concepts.
        dataset_dir : le dossier de la base, ex. outputs/<nom_base>_<id>/ (celui de variables.json).

    Renvoie :
        le chemin du fichier écrit.
    """
    # Identifiant du prompt système
    prompt_id = hashlib.sha1(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:8]

    # Contenu : en-tête (modèle, prompt) puis les concepts
    output = {
        "model": MODEL,
        "system_prompt": SYSTEM_PROMPT,
        "concepts": concepts,
    }

    out_path = Path(dataset_dir) / f"concepts_{prompt_id}.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    # Test manuel : lit un variables.json produit par l'étape 1, puis interroge Qwen
    if len(sys.argv) != 2:
        sys.exit("Usage : python src/bdd/extract_concepts.py <chemin_de_variables.json>")
    variables_path = Path(sys.argv[1])
    data = json.loads(variables_path.read_text(encoding="utf-8"))

    concepts = extract_concepts(data["variables"], data.get("readme"))

    # Une ligne par variable : concept proposé, catégorie, statut
    print()
    for item in concepts:
        concept = item["concept"] or {}
        print(f"{item['variable']:25} | {concept.get('concept_proposed')} | {concept.get('category')} | {concept.get('status')}")

    out_path = save_concepts(concepts, variables_path.parent)
    print(f"\n{len(concepts)} concepts enregistrés dans '{out_path}'")
