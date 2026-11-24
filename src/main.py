"""Point d'entrée unique du projet : des données rangées dans data/ aux concepts SNOMED CT.

Les données sont rangées par type, un dossier par type :
  data/bdd/            bases de données (CSV, Excel)   → extraction src/bdd/
  data/questionnaire/  questionnaires                  → pas encore branché
  data/texte/          textes cliniques                → pas encore branché
Pour chaque fichier : extraction de son type → JSON commun → vérification commune (SNOMED CT).
Les résultats sont écrits dans outputs/ (jamais versionné).

Lancement, depuis la racine du projet, avec Ollama ouvert :
    python -m src.main                                     tous les fichiers de data/
    python -m src.main "data/bdd/ma_base.csv"              un seul fichier (type déduit du dossier)
    python -m src.main "ailleurs/ma_base.csv" --type bdd   un fichier rangé ailleurs
"""

import argparse
from pathlib import Path

from src.bdd.extract_variables import load_readme
from src.bdd.extraction_bdd import run_bdd
from src.verification.verification import run_verification

DATA_DIR = Path("data")                            # dossier des données (chemin relatif à la racine)
DATA_TYPES = ["bdd", "questionnaire", "texte"]     # un sous-dossier de data/ par type de données
BDD_EXTENSIONS = [".csv", ".xlsx", ".xlsm"]        # fichiers lus comme des bases de données


def detect_type(file_path: str | Path) -> str | None:
    """Renvoie le type d'un fichier d'après le dossier où il est rangé, ex. data/bdd/x.csv → "bdd".

    Renvoie None si le fichier n'est pas dans un des dossiers de DATA_TYPES.
    """
    folder = Path(file_path).parent.name
    if folder in DATA_TYPES:
        return folder
    return None


def list_data_files() -> list[tuple[Path, str]]:
    """Liste tous les fichiers à traiter dans data/, avec leur type : [(chemin, type), ...].

    Les fichiers temporaires d'Excel (commençant par "~$", créés quand un fichier est ouvert)
    sont ignorés.
    """
    files = []
    for data_type in DATA_TYPES:
        folder = DATA_DIR / data_type
        if not folder.is_dir():
            continue
        for file_path in sorted(folder.iterdir()):
            if file_path.is_file() and not file_path.name.startswith("~$"):
                files.append((file_path, data_type))
    return files


def process_file(file_path: str | Path, data_type: str) -> dict | None:
    """Traite un fichier : extraction de son type, puis vérification.

    Renvoie le résultat de run_verification ({"verification_path", "experts_path", "counts"}),
    ou None si le type de données n'est pas encore branché ou si le fichier n'est pas lisible.
    """
    if data_type == "bdd":
        if Path(file_path).suffix.lower() not in BDD_EXTENSIONS:
            print(f"-> Ignoré (pas un CSV ni un Excel) : {file_path}")
            return None
        # Un Excel avec un onglet README (comme les fichiers BIO) a ses libellés en ligne 2
        label_row = load_readme(file_path) is not None
        concepts_path = run_bdd(file_path, label_row)
    else:
        print(f"-> Type '{data_type}' pas encore branché, ignoré : {file_path}")
        return None

    return run_verification(concepts_path)


def main() -> None:
    """Commande : traite un fichier (s'il est donné) ou tous les fichiers de data/, puis fait un bilan."""
    parser = argparse.ArgumentParser(description="Pipeline complète : données → concepts SNOMED CT")
    parser.add_argument("file", nargs="?", help="un seul fichier à traiter (sinon : tout data/)")
    parser.add_argument("--type", choices=DATA_TYPES,
                        help="type du fichier, s'il n'est pas rangé dans data/<type>/")
    args = parser.parse_args()

    # Liste des fichiers à traiter, avec leur type
    if args.file is not None:
        data_type = args.type
        if data_type is None:
            data_type = detect_type(args.file)
        if data_type is None:
            raise SystemExit(f"Type inconnu pour {args.file} : rangez-le dans data/<type>/ "
                             f"ou précisez --type ({', '.join(DATA_TYPES)})")
        files = [(Path(args.file), data_type)]
    else:
        files = list_data_files()
        print(f"{len(files)} fichier(s) trouvé(s) dans {DATA_DIR}/")

    # Traitement de chaque fichier ; une erreur sur un fichier n'arrête pas les suivants
    done = []
    failed = []
    for file_path, data_type in files:
        print(f"\n===== {file_path} ({data_type}) =====")
        try:
            result = process_file(file_path, data_type)
        except Exception as error:
            print(f"-> ERREUR : {error}")
            failed.append(file_path)
            continue
        if result is not None:
            done.append((file_path, result))

    # Bilan
    print("\n===== Bilan =====")
    for file_path, result in done:
        print(f"{file_path} -> {result['counts']}")
        print(f"   fichier des experts : {result['experts_path']}")
    for file_path in failed:
        print(f"{file_path} -> ERREUR (voir plus haut)")
    print(f"{len(done)} fichier(s) traité(s), {len(failed)} en erreur.")


if __name__ == "__main__":
    main()
