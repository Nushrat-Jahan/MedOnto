"""Étape 1 du pipeline : extraction des variables d'une base de données structurée.

Ce module doit fonctionner avec n'importe quelle base fournie : aucun nom de colonne,
séparateur ou encodage n'est codé en dur, tout est détecté à partir du fichier lui-même.

Test rapide, depuis la racine du projet :
    python src/extract_variables.py "data/<ma_base>.csv"
"""

import csv
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


if __name__ == "__main__":
    # Test manuel : charge la base passée en argument et affiche un aperçu
    if len(sys.argv) != 2:
        sys.exit("Usage : python src/extract_variables.py <chemin_de_la_base>")
    df = load_dataset(sys.argv[1])
    print(f"{df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(df.head())
