"""Vérification commune : les concepts proposés par le LLM sont-ils dans SNOMED CT ?

Ce fichier est commun à toutes les parties du projet. Il lit un ou plusieurs fichiers au format
JSON commun ({"<source>": [concept, ...]}) et vérifie chaque "concept_proposed" dans SNOMED CT,
avec l'API BioPortal.

Il contient deux familles de fonctions :
- les fonctions de la première version (check_concept_in_snomed, filtre sémantique, main) :
  recherche large, résultat 0 ou 1 par concept ;
- les fonctions de la nouvelle vérification, ajoutées étape par étape (search_exact...) :
  recherche exacte, qui garde l'identifiant SNOMED CT trouvé.

La clé de l'API est lue dans le fichier .env à la racine du projet (BIOPORTAL_API_KEY=...).

Lancement de la première version, depuis la racine du projet :
    python -m src.verification --data "outputs/<base>_<id>/concepts_<id>.json"
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count
import time
import urllib.error
import urllib.parse
import urllib.request
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from src.postprocess import _get_model
import torch
import torch.nn.functional as F

load_dotenv()  # lit le fichier .env (clé de l'API)

# Réglages de la recherche dans l'API BioPortal
DEFAULT_CONFIG = {
    "api_url": "https://data.bioontology.org/search",  # adresse de la recherche
    "ontology": "SNOMEDCT",                             # on ne cherche que dans SNOMED CT
    "pagesize": 1,                                      # nombre de résultats demandés
    "timeout": 30,                                      # secondes d'attente maximum
    "max_retries": 3,                                   # essais en cas d'erreur passagère
    "num_workers": cpu_count()                          # recherches lancées en parallèle
}


# Les 19 grandes branches de SNOMED CT (le premier niveau de l'arbre, sous la racine).
# Point de départ du filtre sémantique : on part de la branche la plus proche du concept.
SNOMED_TOP_LEVEL_CATEGORY: list[dict[str, str]] = [
    {"code": "404684003", "display": "Clinical finding"},
    {"code": "71388000", "display": "Procedure"},
    {"code": "123037004", "display": "Body structure"},
    {"code": "373873003", "display": "Pharmaceutical / biologic product"},
    {"code": "105590001", "display": "Substance"},
    {"code": "243796009", "display": "Situation with explicit context"},
    {"code": "410607006", "display": "Organism"},
    {"code": "260787004", "display": "Physical object"},
    {"code": "363787002", "display": "Observable entity"},
    {"code": "123038009", "display": "Specimen"},
    {"code": "272379006", "display": "Event"},
    {"code": "272391002", "display": "Physical force"},
    {"code": "256235009", "display": "Environment or geographical location"},
    {"code": "410604004", "display": "Social context"},
    {"code": "900000000000441003", "display": "SNOMED CT Model Component"},
    {"code": "370115009", "display": "Special concept"},
    {"code": "254291000", "display": "Staging and scales"},
    {"code": "419891008", "display": "Record artifact"},
    {"code": "106237007", "display": "Qualifier value"}
]

# Mémoire des enfants déjà demandés : {identifiant SNOMED CT: liste de ses enfants}.
# Évite de redemander les mêmes enfants au serveur.
_CHILDREN_CACHE: dict[str, list[dict[str, str]]] = {}


def get_api_key() -> str:
    """Lit la clé de l'API BioPortal dans le fichier .env.

    Arrête le programme avec un message clair si la clé est absente.

    Renvoie :
        la clé, sous forme de texte.
    """
    key = os.environ.get("BIOPORTAL_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "BIOPORTAL_API_KEY manquante dans le fichier .env.\n"
            "Veuillez ajouter la ligne suivante dans votre .env :\n"
            "BIOPORTAL_API_KEY=votre_cle_ici"
        )
    return key


def search_exact(term: str, api_key: str) -> dict | None:
    """Cherche un terme dans SNOMED CT en recherche exacte (API BioPortal).

    La recherche exacte ne renvoie un concept que si son nom, ou un de ses synonymes, est
    exactement le terme cherché (les majuscules ne comptent pas). Contrairement à la recherche
    large, elle ne renvoie jamais un concept qui ressemble seulement au terme : un concept
    trouvé ici peut être accepté tel quel.

    On garde l'identifiant SNOMED CT du concept trouvé et son nom officiel : on peut ainsi
    vérifier le résultat, et plus tard relier la variable à SNOMED CT dans l'ontologie.

    Paramètres :
        term    : le terme à chercher, ex. "white blood cell count".
        api_key : la clé de l'API BioPortal (renvoyée par get_api_key).

    Renvoie :
        {"snomed_id": "767002", "label": "White blood cell count"} si un concept est trouvé ;
        None sinon.
    """
    # Adresse de la recherche : le terme, l'ontologie SNOMED CT, et la recherche exacte
    params = {
        "q": term,
        "ontologies": DEFAULT_CONFIG["ontology"],
        "require_exact_match": "true",
        "include": "prefLabel",
    }
    url = DEFAULT_CONFIG["api_url"] + "?" + urllib.parse.urlencode(params)

    # Envoi de la requête : la clé passe dans l'en-tête (elle n'apparaît pas dans l'adresse)
    request = urllib.request.Request(url, headers={"Authorization": "apikey token=" + api_key})
    with urllib.request.urlopen(request, timeout=DEFAULT_CONFIG["timeout"]) as response:
        data = json.load(response)

    # Premier résultat qui est un vrai concept SNOMED CT : son adresse finit par son identifiant,
    # ex. "http://purl.bioontology.org/ontology/SNOMEDCT/767002" -> "767002"
    for concept in data.get("collection", []):
        snomed_id = concept["@id"].split("/")[-1]
        if snomed_id.isdigit():
            return {"snomed_id": snomed_id, "label": concept.get("prefLabel")}

    return None


def check_concept_in_snomed(query: str, config: dict) -> int:
    """Première version : dit si l'API renvoie au moins un concept SNOMED CT pour ce terme.

    C'est une recherche LARGE (sans require_exact_match), limitée à 1 résultat : l'API renvoie
    presque toujours un concept qui contient à peu près les mêmes mots, même s'il est faux
    (ex. "date of biological test" -> "Date of positive pregnancy test"). Le concept trouvé
    n'est pas gardé : seul le résultat 0 ou 1 est renvoyé.

    En cas d'erreur passagère (serveur surchargé, réseau), la recherche est réessayée
    (attente de 5 s, puis 10 s).

    Paramètres :
        query  : le terme à chercher.
        config : les réglages (DEFAULT_CONFIG, avec en plus la clé "api_key").

    Renvoie :
        1 si l'API renvoie un concept SNOMED CT, 0 sinon (ou en cas d'erreur).
    """
    # Adresse de la recherche (ici, la clé de l'API est mise dans l'adresse)
    params = urllib.parse.urlencode({
        "q": query,
        "ontologies": config["ontology"],
        "apikey": config["api_key"],
        "pagesize": config["pagesize"],
        "include": "prefLabel",
    })
    req = urllib.request.Request(
        f"{config['api_url']}?{params}",
        headers={"User-Agent": "pipeline-LIRMM-A2D2/1.0"},
    )

    for attempt in range(config["max_retries"]):
        try:
            with urllib.request.urlopen(req, timeout=config["timeout"]) as resp:
                data = json.load(resp)

            # Trouvé dès qu'un résultat est un vrai concept SNOMED CT (adresse finissant par un nombre)
            for concept in data.get("collection", []):
                snomed_id = str(concept.get("@id", "")).rsplit("/", 1)[-1]
                if snomed_id.isdigit():
                    return 1
            return 0

        # Erreur du serveur : on réessaie si elle est passagère (429 = trop de requêtes, 5xx = serveur)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < config["max_retries"] - 1:
                time.sleep(5 * (2 ** attempt))
                continue
            print(f"  ERREUR HTTP {e.code} pour la recherche {query!r}", file=sys.stderr)
            return 0
        # Erreur de réseau ou réponse illisible : on réessaie aussi
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt < config["max_retries"] - 1:
                time.sleep(5 * (2 ** attempt))
                continue
            print(f"  ERREUR {e!r} pour la recherche {query!r}", file=sys.stderr)
            return 0

    return 0


def worker_task(job_data: tuple) -> tuple:
    """Vérifie un seul concept : c'est la tâche confiée à chaque processus parallèle de main.

    Paramètre :
        job_data : (source, item_id, concept, config), préparé par extract_jobs_from_nested_json.

    Renvoie :
        (source, item_id, concept, 0 ou 1).
    """
    dataset_key, iid, concept, config = job_data
    query = concept.replace("_", " ")
    is_in_snomed = check_concept_in_snomed(query, config)
    return dataset_key, iid, concept, is_in_snomed


def extract_jobs_from_nested_json(data: dict, config: dict) -> list:
    """Prépare la liste des concepts à vérifier, à partir d'un fichier au format JSON commun.

    Accepte le format commun {"<source>": [concept, ...]} ou une simple liste de concepts.
    Les concepts sans "concept_proposed" sont ignorés.

    Paramètres :
        data   : le contenu du fichier JSON.
        config : les réglages, recopiés dans chaque tâche.

    Renvoie :
        une liste de tâches (source, item_id, concept, config).
    """
    jobs = []
    # Format commun : un dictionnaire {source: liste de concepts}
    if isinstance(data, dict):
        for dataset_key, items in data.items():
            if isinstance(items, list):
                for it in items:
                    iid = it.get("item_id", "")
                    concept = it.get("concept_proposed", "").strip() if it.get("concept_proposed") else ""
                    if concept:
                        jobs.append((dataset_key, iid, concept, config))
    # Simple liste de concepts, sans source : la source s'appelle "default"
    elif isinstance(data, list):
        for it in data:
            iid = it.get("item_id", "")
            concept = it.get("concept_proposed", "").strip() if it.get("concept_proposed") else ""
            if concept:
                jobs.append(("default", iid, concept, config))
    return jobs


def get_children_fhir(parent_id: str, direct_only: bool = True, limit: int = 50) -> list[dict[str, str]]:
    """Demande les concepts situés sous un concept SNOMED CT (ses « enfants »).

    Utilise le serveur public Snowstorm de SNOMED International (format FHIR), et non
    BioPortal. La question est écrite en ECL, le langage de requête de SNOMED CT.

    ATTENTION : en ECL, "<" donne TOUS les descendants (enfants, petits-enfants...), pas
    seulement les enfants directs (ce serait "<!"). Avec la limite de 50 résultats, la liste
    renvoyée peut donc être un échantillon de descendants de tous niveaux.

    Paramètres :
        parent_id   : l'identifiant SNOMED CT du concept parent, ex. "71388000".
        direct_only : True -> "<" (descendants) ; False -> "<<" (descendants et le concept lui-même).
        limit       : nombre maximum de concepts renvoyés.

    Renvoie :
        une liste [{"code": identifiant, "display": nom}], vide en cas d'erreur.
    """
    url = "https://snowstorm.snomedtools.org/fhir/ValueSet/$expand"
    operator = "<" if direct_only else "<<"
    ecl = f"{operator} {parent_id}"  # ex. "< 71388000" : les descendants du concept 71388000

    params = {
        "url": f"http://snomed.info/sct?fhir_vs=ecl/{ecl}",
        "count": limit
    }
    headers = {"Accept": "application/fhir+json"}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            concepts = data.get("expansion", {}).get("contains", [])
            return [{"code": c["code"], "display": c["display"]} for c in concepts]
    except Exception as e:
        print(f"API Error: {e}")
    return []


def get_children_cached(parent_id: str) -> list[dict[str, str]]:
    """Comme get_children_fhir, mais garde les réponses en mémoire (_CHILDREN_CACHE).

    Un même concept n'est demandé qu'une fois au serveur.
    """
    if parent_id not in _CHILDREN_CACHE:
        _CHILDREN_CACHE[parent_id] = get_children_fhir(parent_id)
    return _CHILDREN_CACHE[parent_id]


def compute_embedding(concept: str | dict[str, str]) -> np.ndarray:
    """Transforme un terme en vecteur de nombres (embedding) avec le modèle Bio_ClinicalBERT.

    Deux termes au sens proche donnent des vecteurs proches : c'est ce qui permet de mesurer
    la ressemblance entre deux concepts. Le vecteur est normalisé (longueur 1), donc le
    produit de deux vecteurs donne directement leur ressemblance (similarité cosinus).

    Paramètre :
        concept : un terme, ou un concept {"code", "display"} (on prend alors son nom).

    Renvoie :
        le vecteur (768 nombres).
    """
    tokenizer, encoder = _get_model()  # modèle Bio_ClinicalBERT, chargé une seule fois

    text = concept if isinstance(concept, str) else concept["display"]
    text = text.replace("_", " ")
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():  # pas d'apprentissage : on utilise seulement le modèle
        outputs = encoder(**inputs)

    # Vecteur du premier jeton ([CLS]), qui résume la phrase, puis normalisation
    cls_rep = outputs.last_hidden_state[:, 0, :]
    normalized_tensor = F.normalize(cls_rep, p=2, dim=1)

    return normalized_tensor.squeeze(0).cpu().numpy()


def compute_embeddings(list_concept: list[dict[str, str]]) -> np.ndarray:
    """Calcule l'embedding de chaque concept d'une liste (voir compute_embedding).

    Renvoie :
        un tableau avec un vecteur par concept (vide si la liste est vide).
    """
    if not list_concept:
        return np.empty((0, 768))
    return np.array([compute_embedding(concept) for concept in list_concept])


def give_index_more_similar(embedding: np.ndarray, list_embeddings: np.ndarray) -> tuple[float, int]:
    """Trouve, dans une liste de vecteurs, celui qui ressemble le plus à un vecteur donné.

    Paramètres :
        embedding       : le vecteur du concept cherché.
        list_embeddings : les vecteurs des candidats.

    Renvoie :
        (score de ressemblance du meilleur candidat, entre -1 et 1 ; sa position dans la liste).
    """
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    # Ressemblance avec chaque candidat (produit de vecteurs normalisés = similarité cosinus)
    similarity_scores = (embedding @ list_embeddings.T)[0]
    index = int(np.argmax(similarity_scores))  # position du score le plus élevé
    score = float(similarity_scores[index])
    return score, index


def search_similar_concepts(
    concept: str | dict[str, str],
    list_categorys: list[dict[str, str]],
    categorys_embeddings: np.ndarray | None = None,
    threshold: float = 0.90,
) -> bool:
    """Filtre sémantique : descend dans l'arbre de SNOMED CT vers le concept le plus ressemblant.

    1. On part des grandes branches (list_categorys) et on choisit la plus ressemblante.
    2. On regarde ses enfants : si l'un ressemble plus au concept, on descend vers lui.
    3. On recommence jusqu'à ce qu'aucun enfant ne ressemble plus que le concept actuel.
    Le concept est accepté si la ressemblance finale atteint le seuil.

    Le concept SNOMED CT atteint n'est pas renvoyé : seulement True ou False.

    Paramètres :
        concept              : le terme proposé par le LLM.
        list_categorys       : les concepts de départ (en pratique SNOMED_TOP_LEVEL_CATEGORY).
        categorys_embeddings : leurs vecteurs, s'ils sont déjà calculés (sinon calculés ici).
        threshold            : ressemblance minimum pour accepter le concept (0.90).

    Renvoie :
        True si un concept SNOMED CT assez ressemblant a été atteint, False sinon.
    """
    concept_embedding = compute_embedding(concept)
    if categorys_embeddings is None:
        categorys_embeddings = compute_embeddings(list_categorys)

    # 1. Branche de départ : la plus ressemblante
    top_score, index = give_index_more_similar(concept_embedding, categorys_embeddings)
    current_best_concept = list_categorys[index]

    # 2-3. Descente dans l'arbre tant qu'un enfant ressemble plus
    while True:
        childs = get_children_cached(current_best_concept["code"])

        if not childs:  # plus d'enfants : on s'arrête
            break

        childs_embeddings = compute_embeddings(childs)
        child_top_score, child_index = give_index_more_similar(concept_embedding, childs_embeddings)

        if child_top_score > top_score:
            top_score = child_top_score
            current_best_concept = childs[child_index]
        else:
            break

    return top_score >= threshold


def filter_non_found_snomed_df(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """Applique le filtre sémantique (search_similar_concepts) aux concepts non trouvés.

    Paramètres :
        df        : le tableau des concepts non trouvés par check_concept_in_snomed
                    (colonne "Concept (A2)").
        threshold : ressemblance minimum pour accepter un concept.

    Renvoie :
        les lignes du tableau dont le concept passe le filtre.
    """
    if df.empty:
        return df

    # Vecteurs des 19 grandes branches, calculés une seule fois pour tous les concepts
    categorys_embeddings = compute_embeddings(SNOMED_TOP_LEVEL_CATEGORY)
    mask = df["Concept (A2)"].apply(
        lambda c: search_similar_concepts(
            c,
            SNOMED_TOP_LEVEL_CATEGORY,
            categorys_embeddings,
            threshold,
        )
    )
    return df[mask]


def main() -> None:
    """Première version de la vérification, lancée en ligne de commande.

    1. Lit les fichiers JSON donnés avec --data et prépare un concept à vérifier par ligne.
    2. Vérifie tous les concepts en parallèle (check_concept_in_snomed : recherche large).
    3. Applique le filtre sémantique aux concepts non trouvés.
    4. Affiche le tableau des résultats et le nombre de concepts trouvés.
    Rien n'est écrit dans un fichier.
    """
    # Argument : un ou plusieurs fichiers au format JSON commun
    parser = argparse.ArgumentParser(description="Biomedical Information Extraction Pipeline")
    parser.add_argument("--data", nargs="*", help="Path to zero or more input files (JSON)")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["api_key"] = get_api_key()

    # 1. Un concept à vérifier par ligne, pour tous les fichiers
    all_jobs = []

    if args.data:
        for file_path in args.data:
            if os.path.isfile(file_path):
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                    jobs = extract_jobs_from_nested_json(data, config)
                    all_jobs.extend(jobs)

    if not all_jobs:
        print("Aucun concept valide trouvé dans le(s) fichier(s) spécifié(s).")
        return

    # 2. Vérification en parallèle : un processus par cœur du processeur
    with Pool(processes=config["num_workers"]) as pool:
        results = pool.map(worker_task, all_jobs)

    # Tableau des résultats : une ligne par concept
    rows = []
    for dataset_key, iid, concept, is_in_snomed in results:
        rows.append({
            "Source Dataset": dataset_key,
            "Item ID": iid,
            "Concept (A2)": concept,
            "Is in SNOMED": is_in_snomed
        })

    df = pd.DataFrame(rows)
    n = len(df)
    n_in_snomed = int((df["Is in SNOMED"] == 1).sum())

    # 3. Filtre sémantique sur les concepts non trouvés
    df_not_in_snomed = df[df["Is in SNOMED"] == 0]
    df_filter_not_in_snomed = filter_non_found_snomed_df(df_not_in_snomed)
    n_in_snomed_with_filter = n_in_snomed + len(df_filter_not_in_snomed)

    # 4. Affichage (les 10 premières lignes et les totaux)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("\n--- DataFrame Result (df.head()) ---")
    print(df.head(10))
    print(
        f"\nTotal: {n} | Trouvés dans SNOMED: {n_in_snomed} "
        f"| Avec filtre sémantique: {n_in_snomed_with_filter}"
    )


if __name__ == "__main__":
    main()








if __name__ == "__main__":
    main()