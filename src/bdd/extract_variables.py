"""Étape 1 du pipeline : extraction des variables d'une base de données structurée.

Ce module doit fonctionner avec n'importe quelle base fournie : aucun nom de colonne,
séparateur ou encodage n'est codé en dur, tout est détecté à partir du fichier lui-même.

Test rapide, depuis la racine du projet :
    python src/bdd/extract_variables.py "data/<ma_base>.csv"   (ou .xlsx)
"""

import csv
import hashlib
import json
import sys
from pathlib import Path
import pandas as pd

ENCODINGS = ["utf-8-sig", "latin-1"]  # UTF-8 (avec BOM Excel), puis Latin-1 en dernier recours
SEPARATORS = ",;\t|"  # séparateurs autorisés : virgule, point-virgule, tabulation, barre
SAMPLE_SIZE = 64 * 1024  # extrait de 64 Ko : suffisant pour détecter encodage et séparateur
MAX_CATEGORIES = 20  # au-delà, on ne liste plus les valeurs possibles d'une colonne
OUT_DIR = Path("outputs")  # dossier racine des résultats (chemin relatif à la racine du projet)
EXCEL_EXTENSIONS = (".xlsx", ".xlsm")  # formats Excel lus avec openpyxl

README_SHEET = "README"  # nom de la feuille de documentation d'un fichier Excel (s'il y en a une)

# Repères lus dans les libellés, utilisés pour typer une variable quand il n'y a pas de données
YES_NO_MARK = "(ON)"  # libellé finissant par (ON) : variable Oui/Non
YES_NO_VALUES = ["Non", "Oui"]  # valeurs d'une variable Oui/Non
DATE_WORD = "date"  # libellé contenant le mot "date" : variable date


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Lit une base CSV ou Excel quelconque et la renvoie sous forme de DataFrame pandas.

    Le format est choisi d'après l'extension du fichier :
    - Excel (.xlsx, .xlsm) : seule la première feuille est lue, c'est celle qui contient
      les données (les feuilles suivantes, comme un README, sont ignorées). La première
      ligne donne les noms des colonnes. Nécessite la bibliothèque openpyxl ;
    - sinon, le fichier est lu comme un CSV : le séparateur et l'encodage sont détectés
      automatiquement.
    La base est renvoyée telle quelle : aucune colonne n'est supprimée ni modifiée.
    Si le fichier a une deuxième ligne d'en-tête (libellés des variables), elle apparaît
    comme première ligne du DataFrame : elle n'est pas traitée ici.

    Paramètre :
        path : chemin du fichier CSV ou Excel (texte ou objet Path).

    Renvoie :
        un DataFrame avec une ligne par enregistrement et une colonne par variable.

    Lève :
        FileNotFoundError : si le fichier n'existe pas ;
        ValueError : si l'encodage ou le séparateur d'un CSV n'est pas reconnu.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : '{path}'")

    # Excel : lecture directe de la première feuille
    if path.suffix.lower() in EXCEL_EXTENSIONS:
        return pd.read_excel(path, sheet_name=0)

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


def load_readme(path: str | Path) -> list[dict] | None:
    """Lit la feuille de documentation (README) d'un fichier Excel, s'il en a une.

    Le script ne cherche pas à comprendre ce texte : il le recopie tel quel, pour le
    transmettre plus tard à Qwen comme contexte (codes Oui/Non, format des dates…).

    Paramètre :
        path : chemin du fichier de la base.

    Renvoie :
        une liste avec un dictionnaire par ligne du README, ex.
        {"Topic": "Codes", "Detail": "CSV and Excel use Oui/Non ..."} ;
        None si le fichier n'est pas un Excel ou n'a pas de feuille README.
    """
    path = Path(path)
    if path.suffix.lower() not in EXCEL_EXTENSIONS:
        return None

    # On cherche une feuille nommée README (majuscules ou minuscules)
    for sheet in pd.ExcelFile(path).sheet_names:
        if sheet.lower() == README_SHEET.lower():
            readme = pd.read_excel(path, sheet_name=sheet)
            return readme.to_dict(orient="records")

    return None


def extract_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Retire la ligne des libellés (ligne 2 du fichier) et renvoie les libellés à part.

    À utiliser seulement quand le fichier a une ligne de libellés (option --label-row).
    Exemple : pour la colonne BIO_CRP, le libellé est "Bio. : CRP (mg/L)" et l'unité
    "mg/L" (le texte entre parenthèses à la fin du libellé).

    Paramètre :
        df : la base chargée par load_dataset.

    Renvoie :
        - le tableau des données, sans la ligne des libellés ;
        - un dictionnaire {nom de la colonne: {"label": libellé, "unit": unité}}.
    """
    labels = {}

    for name in df.columns:
        # Le libellé est dans la première ligne du tableau
        label = str(df[name].iloc[0])

        # L'unité est le texte entre parenthèses à la fin du libellé
        unit = None
        if label.endswith(")"):
            start = label.rfind("(")
            unit = label[start + 1 : -1]

        labels[name] = {"label": label, "unit": unit}

    # On enlève la première ligne : il ne reste que les données
    # infer_objects : pandas redevine le type des colonnes (la ligne de texte retirée
    # faisait passer les colonnes de nombres pour du texte)
    data = df.iloc[1:].infer_objects()
    return data, labels


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


def extract_variables(df: pd.DataFrame, labels: dict | None = None) -> list[dict]:
    """Décrit chaque colonne de la base : nom, libellé, unité, type, exemple et valeurs possibles.

    Si des libellés sont fournis (renvoyés par extract_labels), ils sont ajoutés à chaque
    variable avec leur unité.

    Quand une colonne n'a aucune donnée (ex. modèle vide) et que les libellés sont fournis,
    le type est déduit du libellé :
    - libellé contenant "identifier"         -> "identifier" ;
    - libellé finissant par YES_NO_MARK (ON) -> "binary text", valeurs Non / Oui ;
    - libellé contenant le mot DATE_WORD     -> "date" ;
    - libellé avec une unité, ex. (mg/L)     -> "numeric" (nombre, entier ou décimal) ;
    - sinon                                  -> "empty".

    Sinon, le type est déduit des données elles-mêmes, dans cet ordre :
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

    Paramètres :
        df     : la base chargée par load_dataset.
        labels : les libellés renvoyés par extract_labels, ou None s'il n'y en a pas.

    Renvoie :
        une liste avec un dictionnaire par colonne, dans l'ordre de la base :
        {"name", "label", "unit", "type", "example", "n_distinct", "values", "range"}
        (les champs valent None quand ils ne s'appliquent pas).
    """
    variables = []

    for name in df.columns:
        column = df[name]
        non_missing = column.dropna()
        n_distinct = non_missing.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(column)

        # Libellé et unité, s'ils sont fournis
        label = None
        unit = None
        if labels is not None:
            label = labels[name]["label"]
            unit = labels[name]["unit"]

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

        # 5. Pas de données mais un libellé : type déduit du libellé
        if n_distinct == 0 and label is not None:
            if "identifier" in label.lower():
                var_type = "identifier"
            elif label.endswith(YES_NO_MARK):
                var_type = "binary text"
                values = YES_NO_VALUES
                unit = None  # "(ON)" est un codage Oui/Non, pas une unité
            elif DATE_WORD in label.lower().split():
                var_type = "date"
            elif unit is not None:
                var_type = "numeric"

        variables.append({
            "name": str(name),
            "label": label,
            "unit": unit,
            "type": var_type,
            "example": example,
            "n_distinct": int(n_distinct),
            "values": values,
            "range": value_range,
        })

    return variables


def save_variables(variables: list[dict], dataset_path: str | Path, ds_id: str,
                   readme: list[dict] | None = None) -> Path:
    """Enregistre les variables extraites dans outputs/<nom_base>_<id>/variables.json.

    Chaque base a son propre dossier de résultats, nommé d'après le nom du fichier
    (sans extension, espaces remplacés par "_") suivi de son identifiant (dataset_id).
    Le dossier est créé s'il n'existe pas. Si variables.json existe déjà pour cette base,
    il est remplacé : pour une même base, l'extraction donne toujours le même résultat.

    Le fichier contient en en-tête le nom de la base, son identifiant et son README
    (s'il y en a un), puis la liste des variables.

    Paramètres :
        variables    : la liste renvoyée par extract_variables.
        dataset_path : chemin du fichier de la base (sert à nommer le dossier).
        ds_id        : identifiant de la base, renvoyé par dataset_id.
        readme       : la documentation renvoyée par load_readme, ou None.

    Renvoie :
        le chemin du fichier variables.json écrit.
    """
    # Dossier de la base, ex. outputs/The_final_data_after_screening_a4fff73a/
    dataset_name = Path(dataset_path).stem.replace(" ", "_")
    dataset_dir = OUT_DIR / f"{dataset_name}_{ds_id}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Contenu : en-tête (base, ID, README) puis les variables
    output = {
        "dataset": Path(dataset_path).name,
        "dataset_id": ds_id,
        "readme": readme,
        "variables": variables,
    }

    out_path = dataset_dir / "variables.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    # Test manuel : charge la base passée en argument et affiche un aperçu
    if len(sys.argv) != 2:
        sys.exit("Usage : python src/bdd/extract_variables.py <chemin_de_la_base>")
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
    variables = extract_variables(df)
    for variable in variables:
        print(f"{variable['name']:25} | {variable['type']:16} | exemple : {variable['example']} | valeurs : {variable['values']} | intervalle : {variable['range']}")

    # Test de save_variables : écriture du JSON puis relecture pour vérifier
    print()
    out_path = save_variables(variables, sys.argv[1], dataset_id(df))
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"{len(saved['variables'])} variables enregistrées dans '{out_path}'")
