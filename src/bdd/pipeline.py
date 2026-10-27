"""Point d'entrée de la partie bdd : de la base de données structurée au fragment d'ontologie.

Enchaîne les étapes du pipeline dans l'ordre, en appelant les fonctions des modules de src/bdd/.
Les étapes sont ajoutées au fur et à mesure qu'elles sont développées.

Lancement, depuis la racine du projet (même principe que la partie texte, src/text/pipeline.py) :
    python -m src.bdd.pipeline "data/<ma_base>.csv"
    python -m src.bdd.pipeline "data/<ma_base>.xlsx" --label-row   (si la ligne 2 contient les libellés)

"python src/bdd/pipeline.py" ne fonctionne pas : les imports "from src.bdd..." ne sont trouvés
que si Python est lancé depuis la racine du projet avec -m.

Les résultats sont écrits dans outputs/<nom_base>_<id>/.
"""

import argparse
from src.bdd.extract_variables import (load_dataset, load_readme, extract_labels, dataset_id,
                                       extract_variables, save_variables)
from src.bdd.extract_concepts import extract_concepts, save_concepts
from src.bdd.search_snomed import get_api_key, search_concepts, save_candidates
from src.postprocess import postprocess
from src.utils import make_json_from_concepts

# Seuil de ressemblance au-dessus duquel le postprocess commun supprime un concept comme doublon.
# 1.0 : aucune suppression. Dans un tableau, chaque colonne est une variable différente ; avec le
# seuil par défaut (0.85), des examens différents (neutrophiles, lymphocytes...) étaient supprimés.
SIMILARITY_THRESHOLD = 1.0


def main():
    """Lit le chemin de la base passé en argument, puis exécute les étapes du pipeline."""
    # Argument : chemin de la base à traiter
    parser = argparse.ArgumentParser(description="Pipeline : base de données structurée -> ontologie")
    parser.add_argument("dataset", help="chemin du fichier de la base (CSV ou Excel)")
    parser.add_argument("--label-row", action="store_true",
                        help="la ligne 2 du fichier contient les libellés des variables")
    args = parser.parse_args()

    # Clé de l'API SNOMED CT (étape 3), vérifiée dès le début pour ne pas échouer après Qwen
    api_key = get_api_key()

    # Étape 1 : extraction des variables
    print(f"-> Chargement de la base '{args.dataset}'...")
    df = load_dataset(args.dataset)

    # Ligne de libellés (option --label-row) : retirée des données
    labels = None
    if args.label_row:
        df, labels = extract_labels(df)
        print(f"-> {len(labels)} libellés lus sur la ligne 2")

    ds_id = dataset_id(df)
    print(f"-> {df.shape[0]} lignes, {df.shape[1]} colonnes (ID de la base : {ds_id})")

    variables = extract_variables(df, labels)
    # Documentation de la base (feuille README d'un Excel), recopiée dans le JSON
    readme = load_readme(args.dataset)
    if readme is not None:
        print(f"-> README trouvé : {len(readme)} lignes de documentation")

    out_path = save_variables(variables, args.dataset, ds_id, readme)
    print(f"-> Étape 1 : {len(variables)} variables extraites dans '{out_path}'")

    # Étape 2 : un concept proposé par Qwen pour chaque variable
    concepts = extract_concepts(variables, readme)

    # Nettoyage commun des réponses de Qwen (src/postprocess.py), comme dans la partie texte.
    # postprocess reçoit une liste de groupes de réponses : ici un seul groupe, la base.
    raw_outputs = []
    for item in concepts:
        raw_outputs.append(item["raw_output"])
    clean_concepts = postprocess([raw_outputs], threshold=SIMILARITY_THRESHOLD)[0]
    print(f"-> Postprocess : {len(clean_concepts)} concepts gardés sur {len(concepts)}")

    clean_dicts = make_json_from_concepts(clean_concepts)
    concepts_path = save_concepts(clean_dicts, out_path.parent, args.dataset)
    print(f"-> Étape 2 : {len(clean_dicts)} concepts au format commun dans '{concepts_path}'")

    # Étape 3 : recherche des concepts proposés dans SNOMED CT (API BioPortal)
    searches = search_concepts(concepts, api_key)
    candidates_path = save_candidates(searches, concepts_path)
    print(f"-> Étape 3 : candidats SNOMED CT de {len(searches)} variables dans '{candidates_path}'")


if __name__ == "__main__":
    main()
