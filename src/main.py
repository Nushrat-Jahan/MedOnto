"""Point d'entrée unique du projet : des données rangées dans data/ aux concepts SNOMED CT.

Les données sont rangées par type, un dossier par type :
  data/bdd/            bases de données (CSV, Excel)   → extraction src/bdd/
  data/questionnaire/  questionnaires (.json)          → extraction src/questions/
  data/texte/          textes cliniques (.pdf, .txt)   → extraction src/text/
                       + hf_datasets.txt : des datasets Hugging Face de textes, un par ligne
                         ("nom_du_dataset colonne_du_texte"), lus en direct sur Internet
Pour chaque fichier : extraction de son type → JSON commun → vérification commune (SNOMED CT).
Les résultats sont écrits dans outputs/ (jamais versionné).

Lancement, depuis la racine du projet, avec Ollama ouvert :
    python -m src.main                                     tous les fichiers de data/
    python -m src.main "data/bdd/ma_base.csv"              un seul fichier (type déduit du dossier)
    python -m src.main "ailleurs/ma_base.csv" --type bdd   un fichier rangé ailleurs
    python -m src.main --max-documents 3                   textes lus par dataset Hugging Face (5 par défaut)
"""

import argparse
from pathlib import Path

from src.bdd.extract_variables import load_readme
from src.bdd.extraction_bdd import run_bdd
from src.questions.extraction_questions import run_questions
from src.verification.verification import run_verification

DATA_DIR = Path("data")                            # dossier des données (chemin relatif à la racine)
DATA_TYPES = ["bdd", "questionnaire", "texte"]     # un sous-dossier de data/ par type de données
BDD_EXTENSIONS = [".csv", ".xlsx", ".xlsm"]        # fichiers lus comme des bases de données
QUESTIONNAIRE_EXTENSIONS = [".json"]               # fichiers de questions au format items.json
TEXT_EXTENSIONS = [".pdf", ".txt"]                 # textes cliniques
HF_LIST_FILE = "hf_datasets.txt"                   # dans data/texte/ : la liste des datasets Hugging Face
DEFAULT_MAX_DOCUMENTS = 5                          # textes lus par dataset (chaque texte = un appel à Qwen)


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
    elif data_type == "questionnaire":
        if Path(file_path).suffix.lower() not in QUESTIONNAIRE_EXTENSIONS:
            print(f"-> Ignoré (format non lu, seulement un fichier de questions .json) : {file_path}")
            return None
        concepts_path = run_questions(file_path)
    elif data_type == "texte":
        if Path(file_path).suffix.lower() not in TEXT_EXTENSIONS:
            print(f"-> Ignoré (pas un PDF ni un TXT) : {file_path}")
            return None
        # Importé seulement ici : la partie texte demande des bibliothèques en plus (datasets,
        # gliner). Si elles manquent, seuls les textes échouent, pas les bases ni les questionnaires.
        from src.text.extraction_text import run_text
        concepts_path = run_text(file_path)
        if concepts_path is None:
            print(f"-> Aucun concept extrait du texte : {file_path}")
            return None
    else:
        print(f"-> Type '{data_type}' pas encore branché, ignoré : {file_path}")
        return None

    return run_verification(concepts_path)


def read_hf_list(list_path: str | Path) -> list[tuple[str, str]]:
    """Lit la liste des datasets Hugging Face : une ligne = "nom_du_dataset colonne_du_texte".

    Les lignes vides et celles qui commencent par # sont ignorées.
    Renvoie [(nom_du_dataset, colonne), ...].
    """
    datasets = []
    with open(list_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                print(f"-> Ligne ignorée (attendu : nom_du_dataset colonne) : {line}")
                continue
            datasets.append((parts[0], parts[1]))
    return datasets


def process_hf_dataset(dataset_name: str, column_name: str, max_documents: int) -> dict | None:
    """Traite un dataset Hugging Face de textes : extraction des max_documents premiers textes, puis vérification.

    Renvoie le résultat de run_verification, ou None si aucun concept n'a été extrait.
    """
    # Importé seulement ici, pour la même raison que run_text (bibliothèques en plus)
    from src.text.extraction_text import run_text_hf
    concepts_path = run_text_hf(dataset_name, column_name, max_documents)
    if concepts_path is None:
        print(f"-> Aucun concept extrait du dataset : {dataset_name}")
        return None
    return run_verification(concepts_path)


def main() -> None:
    """Commande : traite un fichier (s'il est donné) ou tous les fichiers de data/, puis fait un bilan."""
    parser = argparse.ArgumentParser(description="Pipeline complète : données → concepts SNOMED CT")
    parser.add_argument("file", nargs="?", help="un seul fichier à traiter (sinon : tout data/)")
    parser.add_argument("--type", choices=DATA_TYPES,
                        help="type du fichier, s'il n'est pas rangé dans data/<type>/")
    parser.add_argument("--max-documents", type=int, default=DEFAULT_MAX_DOCUMENTS,
                        help=f"textes lus par dataset Hugging Face ({DEFAULT_MAX_DOCUMENTS} par défaut)")
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

    # Liste des traitements : un par fichier, et un par dataset de hf_datasets.txt
    jobs = []
    for file_path, data_type in files:
        if data_type == "texte" and Path(file_path).name == HF_LIST_FILE:
            for dataset_name, column_name in read_hf_list(file_path):
                jobs.append((f"{dataset_name} (Hugging Face, colonne {column_name})", "hf",
                             (dataset_name, column_name)))
        else:
            jobs.append((str(file_path), data_type, file_path))

    # Traitement de chacun ; une erreur n'arrête pas les suivants
    done = []
    failed = []
    for label, data_type, source in jobs:
        print(f"\n===== {label} ({data_type}) =====")
        try:
            if data_type == "hf":
                result = process_hf_dataset(source[0], source[1], args.max_documents)
            else:
                result = process_file(source, data_type)
        except Exception as error:
            print(f"-> ERREUR : {error}")
            failed.append(label)
            continue
        if result is not None:
            done.append((label, result))

    # Bilan
    print("\n===== Bilan =====")
    for label, result in done:
        print(f"{label} -> {result['counts']}")
        print(f"   fichier des experts : {result['experts_path']}")
    for label in failed:
        print(f"{label} -> ERREUR (voir plus haut)")
    print(f"{len(done)} traité(s), {len(failed)} en erreur.")


if __name__ == "__main__":
    main()
