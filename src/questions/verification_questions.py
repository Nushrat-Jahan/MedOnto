# -*- coding: utf-8 -*-
"""
Version séquentielle (UN SEUL processus) de src/verification.py.

Pourquoi : src/verification.py lance un processus par cœur du processeur, et chacun
recharge torch. Sur un ordinateur avec peu de mémoire, Windows renvoie alors
« WinError 1455 : le fichier de pagination est insuffisant ».

Ce script réutilise TELLES QUELLES les fonctions du groupe (recherche BioPortal,
filtre sémantique) ; seule la boucle parallèle est remplacée par une boucle simple.
src/verification.py n'est pas modifié.

Entrée  : un ou plusieurs JSON via --data (comme src/verification.py) ;
          sans --data, res/questions/concepts_questions.json
Sorties : bilan affiché + res/questions/verification_snomed.csv

À lancer depuis la RACINE du dépôt :
    python -m src.questions.verification_questions
    python -m src.questions.verification_questions --data fichier1.json fichier2.json

Sur un ordinateur avec assez de mémoire, le script commun src/verification.py
(mêmes arguments --data) donne les mêmes résultats.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.utils import load_json_file
from src.verification import (
    DEFAULT_CONFIG,
    check_concept_in_snomed,
    extract_jobs_from_nested_json,
    filter_non_found_snomed_df,
    get_api_key,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "res" / "questions" / "concepts_questions.json"
OUT_CSV = ROOT / "res" / "questions" / "verification_snomed.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Vérification SNOMED (un seul processus)")
    parser.add_argument("--data", nargs="*",
                        help="Un ou plusieurs JSON de concepts extraits "
                             "(par défaut : res/questions/concepts_questions.json)")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["api_key"] = get_api_key()

    jobs = []
    for path in (args.data or [str(DEFAULT_DATA)]):
        if not Path(path).is_file():
            print(f"Fichier introuvable, ignoré : {path}")
            continue
        jobs.extend(extract_jobs_from_nested_json(load_json_file(path), config))
    if not jobs:
        print("Aucun concept valide trouvé dans le(s) fichier(s) spécifié(s).")
        return

    rows = []
    for i, (dataset_key, iid, concept, cfg) in enumerate(jobs, 1):
        found = check_concept_in_snomed(concept.replace("_", " "), cfg)
        print(f"[{i}/{len(jobs)}] {iid} ({concept}) : {'trouvé' if found else 'non trouvé'}")
        rows.append({
            "Source Dataset": dataset_key,
            "Item ID": iid,
            "Concept (A2)": concept,
            "Is in SNOMED": found,
        })

    df = pd.DataFrame(rows)
    n = len(df)
    n_found = int((df["Is in SNOMED"] == 1).sum())

    # Filtre sémantique du groupe, appliqué seulement aux concepts non trouvés
    df_not_found = df[df["Is in SNOMED"] == 0]
    df_pass = filter_non_found_snomed_df(df_not_found)
    passed = set(df_pass.index)
    df["Semantic filter"] = [
        "n/a" if row["Is in SNOMED"] == 1 else ("1" if idx in passed else "0")
        for idx, row in df.iterrows()
    ]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print("\n--- Résultat ---")
    print(df.to_string(index=False))
    print(f"\nTotal: {n} | Trouvés dans SNOMED: {n_found} "
          f"| Avec filtre sémantique: {n_found + len(df_pass)}")
    print(f"CSV : {OUT_CSV}")


if __name__ == "__main__":
    main()