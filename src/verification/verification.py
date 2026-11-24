"""Point d'entrée de la vérification commune : du JSON commun aux concepts SNOMED CT.

Enchaîne les étapes de la vérification dans l'ordre, en appelant les fonctions de
src/verification/snomed_check.py :
  étapes 1-2 : recherche exacte du concept, puis de ses synonymes (verify_exact) ;
  étape 3    : l'API propose des concepts, Qwen choisit (verify_llm).
Deux fichiers sont écrits à côté du fichier des concepts :
  concepts_<id>_verification.json : le résultat complet, pour tous les concepts ;
  concepts_<id>_experts.xlsx      : les concepts à examiner par les experts (onglets CONCEPTS et README).

Lancement, depuis la racine du projet, avec Ollama ouvert :
    python -m src.verification.verification "outputs/<nom_base>_<id>/concepts_<id>.json"
"""

import argparse
import json
from pathlib import Path

from src.verification.snomed_api import get_api_key
from src.verification.snomed_check import save_candidates, save_verification, verify_exact, verify_llm


def run_verification(concepts_path: str | Path) -> dict:
    """Vérifie un fichier au format JSON commun dans SNOMED CT, puis affiche un résumé.

    Appelable par un autre programme (comme src/main.py) ou par la commande de ce fichier.

    Paramètre :
        concepts_path : le fichier JSON commun, ex. outputs/<base>_<id>/concepts_<id>.json.

    Renvoie :
        {"verification_path": chemin du _verification.json,
         "experts_path": chemin du _experts.xlsx,
         "counts": nombre de concepts par valeur de found_by, ex. {"concept": 8, "llm_choice": 10}}

    Lève :
        FileNotFoundError : si le fichier n'existe pas.
    """
    concepts_path = Path(concepts_path)
    if not concepts_path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {concepts_path}")

    with open(concepts_path, encoding="utf-8") as f:
        data = json.load(f)

    # Une simple liste de concepts (sans source) est rangée sous la source "default"
    if isinstance(data, list):
        data = {"default": data}

    api_key = get_api_key()

    print("Étapes 1-2 : recherche exacte du concept, puis de ses synonymes")
    results = verify_exact(data, api_key)

    print("Étape 3 : l'API propose des concepts, Qwen choisit")
    results = verify_llm(results, api_key)

    verification_path = save_verification(results, concepts_path)
    experts_path = save_candidates(results, concepts_path)  # fichier Excel pour les experts

    # Résumé : nombre de concepts pour chaque façon d'être trouvé
    counts = {}
    total = 0
    for source, concepts in results.items():
        for concept in concepts:
            found_by = concept["found_by"]
            if found_by not in counts:
                counts[found_by] = 0
            counts[found_by] = counts[found_by] + 1
            total = total + 1

    print(f"\nRésumé ({total} concepts) :")
    for found_by, count in counts.items():
        print(f"  {found_by:12} : {count}")
    print("Les concepts trouvés par Qwen (llm_...) sont à faire confirmer par les experts.")

    return {
        "verification_path": verification_path,
        "experts_path": experts_path,
        "counts": counts,
    }


def main() -> None:
    """Commande : lit le chemin du fichier JSON commun dans le terminal, puis lance run_verification."""
    parser = argparse.ArgumentParser(description="Vérification commune des concepts dans SNOMED CT")
    parser.add_argument("concepts", help="fichier JSON commun, ex. outputs/<base>_<id>/concepts_<id>.json")
    args = parser.parse_args()

    try:
        run_verification(args.concepts)
    except FileNotFoundError as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
