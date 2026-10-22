"""Étape 2 du pipeline : extraction des concepts par Qwen, à partir des variables extraites.

Pour chaque variable de variables.json (étape 1), le script construit un prompt, l'envoie à
Qwen (en local via Ollama) et récupère sa proposition de concept au format commun du projet :
{"item_id", "concept_proposed", "category", "relation", "evidence", "status"}.
Les variables sont envoyées une par une : un prompt court et une réponse simple, plus fiables
pour un modèle de 7 milliards de paramètres.

Test rapide, depuis la racine du projet :
    python src/bdd/extract_concepts.py "outputs/<nom_base>_<id>/variables.json"
"""

import json
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
