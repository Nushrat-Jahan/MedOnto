"""Point d'entrée du pipeline : de la base de données structurée au fragment d'ontologie.

Enchaîne les étapes du pipeline dans l'ordre, en appelant les fonctions des modules de src/.
Les étapes sont ajoutées au fur et à mesure qu'elles sont développées.

Lancement, depuis la racine du projet :
    python main.py "data/<ma_base>.csv"

Les résultats sont écrits dans outputs/<nom_base>_<id>/.
"""

import argparse
from src.bdd.extract_variables import load_dataset, dataset_id, extract_variables, save_variables


def main():
    """Lit le chemin de la base passé en argument, puis exécute les étapes du pipeline."""
    # Argument : chemin de la base à traiter
    parser = argparse.ArgumentParser(description="Pipeline : base de données structurée -> ontologie")
    parser.add_argument("dataset", help="chemin du fichier de la base (CSV)")
    args = parser.parse_args()

    # Étape 1 : extraction des variables
    print(f"-> Chargement de la base '{args.dataset}'...")
    df = load_dataset(args.dataset)
    ds_id = dataset_id(df)
    print(f"-> {df.shape[0]} lignes, {df.shape[1]} colonnes (ID de la base : {ds_id})")

    variables = extract_variables(df)
    out_path = save_variables(variables, args.dataset, ds_id)
    print(f"-> Étape 1 : {len(variables)} variables extraites dans '{out_path}'")


if __name__ == "__main__":
    main()
