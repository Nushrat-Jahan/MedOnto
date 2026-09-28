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
        une chaîne de 8 caractères hexadécimaux, par ex. "3f2a9c1b".
    """

    # Noms des colonnes séparés par des virgules, ex. "SEQN,Age,BMI"
    column_names = ",".join(df.columns)

    # Empreinte SHA-1 de ce texte
    fingerprint = hashlib.sha1(column_names.encode("utf-8"))
    return fingerprint.hexdigest()[:8]


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
