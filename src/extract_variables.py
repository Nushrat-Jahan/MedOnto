"""Étape 1 du pipeline : extraction des variables d'une base de données structurée.

Ce module doit fonctionner avec n'importe quelle base fournie : aucun nom de colonne,
séparateur ou encodage n'est codé en dur, tout est détecté à partir du fichier lui-même.

Test rapide, depuis la racine du projet :
    python src/extract_variables.py "data/<ma_base>.csv"
"""

import csv
import hashlib
import sys
from pathlib import Path
import pandas as pd

ENCODINGS = ["utf-8-sig", "latin-1"]  # UTF-8 (avec BOM Excel), puis Latin-1 en dernier recours
SEPARATORS = ",;\t|"  # séparateurs autorisés : virgule, point-virgule, tabulation, barre
SAMPLE_SIZE = 64 * 1024  # extrait de 64 Ko : suffisant pour détecter encodage et séparateur
MAX_CATEGORIES = 20  # au-delà, on ne liste plus les valeurs possibles d'une colonne


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Lit une base CSV quelconque et la renvoie sous forme de DataFrame pandas.

    Le séparateur et l'encodage sont détectés automatiquement.
    La base est renvoyée telle quelle : aucune colonne n'est supprimée ni modifiée.

    Paramètre :
        path : chemin du fichier CSV (texte ou objet Path).

    Renvoie :
        un DataFrame avec une ligne par enregistrement et une colonne par variable.

    Lève :
        FileNotFoundError : si le fichier n'existe pas ;
        ValueError : si l'encodage ou le séparateur n'est pas reconnu.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : '{path}'")

    # 1. Encodage : on garde le premier qui lit l'extrait sans erreur
    for encoding in ENCODINGS:
        try:
            with path.open(encoding=encoding, newline="") as f:
                sample = f.read(SAMPLE_SIZE)
            break
        except UnicodeDecodeError:
            continue
    else:  # aucun encodage n'a fonctionné
        raise ValueError(f"Encodage non reconnu pour '{path}' (essayés : {ENCODINGS})")

    # 2. Séparateur : le Sniffer cherche le caractère présent autant de fois sur chaque ligne
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=SEPARATORS).delimiter
    except csv.Error:
        raise ValueError(f"Séparateur non reconnu pour '{path}' (essayés : {SEPARATORS!r})")

    # 3. Lecture complète de la base
    return pd.read_csv(path, sep=sep, encoding=encoding)


def dataset_id(df: pd.DataFrame) -> str:
    """Calcule un identifiant court et stable pour une base, à partir des noms de ses colonnes.

    L'identifiant est l'empreinte SHA-1 de la liste des noms de colonnes, dans l'ordre,
    tronquée à 8 caractères :
    - mêmes colonnes -> même ID, même si le fichier est renommé ou déplacé ;
    - une colonne ajoutée, retirée, renommée ou déplacée -> ID différent.
    Les valeurs des lignes ne sont pas prises en compte : deux bases avec les mêmes
    colonnes mais des données différentes auront le même ID.
    Il sert à ranger les résultats de chaque base dans son propre dossier de sortie.

    Paramètre :
        df : la base déjà chargée par load_dataset (le fichier n'est pas relu).

    Renvoie :
        une chaîne de 8 caractères hexadécimaux.
    """

    # Noms des colonnes séparés par des virgules, ex. "SEQN,Age,BMI"
    column_names = ",".join(df.columns)

    # Empreinte SHA-1 de ce texte
    fingerprint = hashlib.sha1(column_names.encode("utf-8"))
    return fingerprint.hexdigest()[:8]


def extract_variables(df: pd.DataFrame) -> list[dict]:
    """Décrit chaque colonne de la base : nom, type, exemple et valeurs possibles.

    Le type est déduit des données elles-mêmes, dans cet ordre :
    - "empty"            : colonne sans aucune valeur ;
    - "identifier"       : numéro qui distingue chaque ligne, comme un numéro de patient
                           (ex. SEQN). Reconnu si les 3 conditions sont réunies :
                           toutes les valeurs sont différentes, aucune ne manque, et aucune
                           n'a de décimales (pour ne pas confondre avec une mesure comme BRI).
                           Ce n'est pas une variable médicale : elle ne sera pas envoyée à Qwen ;
    - "binary numeric"   : exactement 2 valeurs numériques (ex. 0 / 1) ;
    - "binary text"      : exactement 2 valeurs textuelles (ex. Normal / Hypertension) ;
    - "decimal"          : nombres avec décimales (ex. BMI) ;
    - "integer"          : nombres entiers (ex. Age) ;
    - "categorical text" : texte avec au plus MAX_CATEGORIES valeurs différentes ;
    - "free text"        : texte avec plus de MAX_CATEGORIES valeurs différentes.

    Les valeurs possibles ne sont listées que s'il y en a au plus MAX_CATEGORIES :
    c'est le cas des variables catégorielles et des variables codées (ex. 0 / 1).
    Pour les nombres trop nombreux pour être listés (ex. BMI, Age), on donne à la place
    l'intervalle des valeurs, rédigé en anglais pour être compris tel quel par Qwen :
    "any value between 14.1 and 82.0".

    Paramètre :
        df : la base chargée par load_dataset.

    Renvoie :
        une liste avec un dictionnaire par colonne, dans l'ordre de la base :
        {"name", "type", "example", "n_distinct", "values", "range"}
        ("example", "values" et "range" valent None quand ils ne s'appliquent pas).
    """
    variables = []

    for name in df.columns:
        column = df[name]
        non_missing = column.dropna()
        n_distinct = non_missing.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(column)

        # Décimales : au moins une valeur numérique qui n'est pas un nombre entier
        is_decimal = False
        if is_numeric:
            is_decimal = (non_missing % 1 != 0).any()

        # 1. Type de la variable
        if n_distinct == 0:
            var_type = "empty"
        elif n_distinct == len(df) and not is_decimal:
            var_type = "identifier"
        elif n_distinct == 2 and is_numeric:
            var_type = "binary numeric"
        elif n_distinct == 2:
            var_type = "binary text"
        elif is_decimal:
            var_type = "decimal"
        elif is_numeric:
            var_type = "integer"
        elif n_distinct <= MAX_CATEGORIES:
            var_type = "categorical text"
        else:
            var_type = "free text"

        # 2. Exemple : première valeur non manquante
        example = None
        if n_distinct > 0:
            example = non_missing.head(1).tolist()[0]

        # 3. Valeurs possibles, seulement s'il y en a peu (et jamais pour un identifiant)
        values = None
        if 0 < n_distinct <= MAX_CATEGORIES and var_type != "identifier":
            values = sorted(non_missing.unique().tolist())

        # 4. Nombres trop nombreux pour être listés : on donne l'intervalle, en anglais pour Qwen
        value_range = None
        if values is None and var_type in ("integer", "decimal"):
            value_range = f"any value between {non_missing.min()} and {non_missing.max()}"

        variables.append({
            "name": str(name),
            "type": var_type,
            "example": example,
            "n_distinct": int(n_distinct),
            "values": values,
            "range": value_range,
        })

    return variables


if __name__ == "__main__":
    # Test manuel : charge la base passée en argument et affiche un aperçu
    if len(sys.argv) != 2:
        sys.exit("Usage : python src/extract_variables.py <chemin_de_la_base>")
    df = load_dataset(sys.argv[1])
    print(f"{df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(df.head())

    # Test de dataset_id
    print()
    print(f"ID de la base : {dataset_id(df)}")

    # Mêmes colonnes mais moins de lignes -> même ID attendu
    first_rows = df.head(10)
    print(f"ID des 10 premières lignes : {dataset_id(first_rows)}")

    # Une colonne en moins -> ID différent attendu
    first_column = df.columns[0]
    without_first_column = df.drop(columns=first_column)
    print(f"ID sans la colonne '{first_column}' : {dataset_id(without_first_column)}")

    # Test de extract_variables : une ligne par variable
    print()
    for variable in extract_variables(df):
        print(f"{variable['name']:25} | {variable['type']:16} | exemple : {variable['example']} | valeurs : {variable['values']} | intervalle : {variable['range']}")
