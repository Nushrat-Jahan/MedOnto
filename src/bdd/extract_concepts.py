"""Étape 2 du pipeline : extraction des concepts par Qwen, à partir des variables extraites.

Pour chaque variable de variables.json (étape 1), le script construit un prompt, l'envoie à
Qwen (en local via Ollama) et récupère sa proposition de concept au format JSON commun du projet :
{"item_id", "concept_proposed", "category", "relation", "evidence", "synonyms"}.
Le prompt système est le même pour toutes les parties du projet : il est lu dans
configs/systems_prompt/system_prompt.txt. Le message de chaque variable est propre à la partie
bdd : c'est configs/systems_prompt/user_template_bdd.txt, où {txt} est remplacé par la
description de la variable.
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

# Prompt système commun à toutes les parties du projet (le même que la partie texte).
# Chemin relatif : les scripts se lancent depuis la racine du projet.
SYSTEM_PROMPT_PATH = Path("configs/systems_prompt/system_prompt.txt")
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

# Message propre à la partie bdd : les règles pour une variable, puis {txt} remplacé par sa description
USER_TEMPLATE_PATH = Path("configs/systems_prompt/user_template_bdd.txt")
USER_TEMPLATE = USER_TEMPLATE_PATH.read_text(encoding="utf-8")
PLACEHOLDER = "{txt}"  # même repère que dans le user_template de la partie texte
if PLACEHOLDER not in USER_TEMPLATE:
    sys.exit(f"{USER_TEMPLATE_PATH} doit contenir le repère {PLACEHOLDER}")

# Catégories autorisées dans la réponse de Qwen (même liste que dans le prompt système commun)
CATEGORIES = ["finding", "symptom", "disorder", "behavior", "observable_entity", "assessment_scale"]


def build_user_prompt(variable: dict, readme: list[dict] | None = None) -> str:
    """Rédige le message envoyé à Qwen pour une variable, à partir de sa fiche (étape 1).

    Le message est le user_template de la partie bdd (règles propres aux variables d'un
    tableau), où le repère {txt} est remplacé par la description de la variable.

    La description contient :
    - la documentation de la base (README), si elle existe, recopiée ligne par ligne ;
    - la description de la variable : nom, libellé, unité, type, valeurs possibles,
      intervalle et exemple. Seuls les champs renseignés sont écrits : une variable sans
      libellé n'a pas de ligne "Label".

    Le nom de la variable est présenté comme son "Item ID" : le prompt système commun demande
    un champ "item_id", qui doit être le nom de la variable pour retrouver d'où vient le concept.

    Exemple de description pour la variable BIO_CRP d'un modèle BIO :
        Item ID (variable name): BIO_CRP
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
    lines.append(f"Item ID (variable name): {variable['name']}")
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

    description = "\n".join(lines)

    # Message envoyé : le user_template, avec la description à la place de {txt}
    return USER_TEMPLATE.replace(PLACEHOLDER, description)


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
    du format demandé (JSON invalide, item_id différent du nom de la variable, catégorie hors
    liste, synonyms qui n'est pas une liste) sont signalées à l'écran, pour être revérifiées
    à la main.

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
            if concept.get("item_id") != variable["name"]:
                print(f"   ATTENTION : item_id différent du nom de la variable -> {concept.get('item_id')}")
            if concept.get("category") not in CATEGORIES:
                print(f"   ATTENTION : catégorie hors liste -> {concept.get('category')}")
            if not isinstance(concept.get("synonyms"), list):
                print(f"   ATTENTION : synonyms n'est pas une liste -> {concept.get('synonyms')}")

        concepts.append({
            "variable": variable["name"],
            "user_prompt": user_prompt,
            "raw_output": raw,
            "concept": concept,
        })

    return concepts


def save_concepts(concepts: list[dict], dataset_dir: str | Path, source: str) -> Path:
    """Enregistre les concepts proposés par Qwen au format JSON commun à tout le projet.

    C'est le même format que la partie texte (src/text/pipeline.py), pour que la vérification
    commune puisse lire les résultats de toutes les parties :
        {"<source>": [concept, concept, ...]}
    où chaque concept est la réponse de Qwen telle quelle :
        {"item_id", "concept_proposed", "category", "relation", "evidence", "synonyms"}.
    Les réponses qui ne sont pas un JSON valide ne sont pas écrites (elles ont déjà été
    signalées par extract_concepts).

    Le fichier s'appelle concepts_<id du prompt>.json, où l'ID est l'empreinte SHA-1 du
    prompt système et du user_template mis bout à bout, réduite à 8 caractères (même principe
    que dataset_id) :
    - mêmes prompts -> même fichier : comme temperature=0, la réponse est la même ;
    - un des deux prompts modifié -> nouveau fichier : les résultats des anciens prompts sont
      conservés, ce qui permet de comparer les versions des prompts.

    Paramètres :
        concepts    : la liste renvoyée par extract_concepts.
        dataset_dir : le dossier de la base, ex. outputs/<nom_base>_<id>/ (celui de variables.json).
        source      : le nom de la base, utilisé comme clé du JSON (ex. "data/ma_base.xlsx").

    Renvoie :
        le chemin du fichier écrit.
    """
    # Identifiant des prompts : prompt système + user_template de la partie bdd
    prompts = SYSTEM_PROMPT + USER_TEMPLATE
    prompt_id = hashlib.sha1(prompts.encode("utf-8")).hexdigest()[:8]

    # On ne garde que les réponses valides de Qwen
    valid_concepts = []
    for item in concepts:
        if item["concept"] is not None:
            valid_concepts.append(item["concept"])

    # Format commun : le nom de la base comme clé, la liste de ses concepts comme valeur
    output = {source: valid_concepts}

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

    # Une ligne par variable : concept proposé, catégorie, synonymes
    print()
    for item in concepts:
        concept = item["concept"]
        if concept is None:
            concept = {}
        print(f"{item['variable']:25} | {concept.get('concept_proposed')} | {concept.get('category')} | {concept.get('synonyms')}")

    out_path = save_concepts(concepts, variables_path.parent, data["dataset"])
    print(f"\n{len(concepts)} concepts enregistrés dans '{out_path}'")
