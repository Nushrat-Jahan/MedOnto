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

load_dotenv()

DEFAULT_CONFIG = {
    "api_url": "https://data.bioontology.org/search",
    "ontology": "SNOMEDCT",
    "pagesize": 1,
    "timeout": 30,
    "max_retries": 3,
    "num_workers": cpu_count()
}


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

_CHILDREN_CACHE: dict[str, list[dict[str, str]]] = {}


def get_api_key() -> str:
    key = os.environ.get("BIOPORTAL_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "BIOPORTAL_API_KEY manquante dans le fichier .env.\n"
            "Veuillez ajouter la ligne suivante dans votre .env :\n"
            "BIOPORTAL_API_KEY=votre_cle_ici"
        )
    return key


def check_concept_in_snomed(query: str, config: dict) -> int:
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

            for concept in data.get("collection", []):
                snomed_id = str(concept.get("@id", "")).rsplit("/", 1)[-1]
                if snomed_id.isdigit():
                    return 1
            return 0

        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < config["max_retries"] - 1:
                time.sleep(5 * (2 ** attempt))
                continue
            print(f"  ERREUR HTTP {e.code} pour la recherche {query!r}", file=sys.stderr)
            return 0
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt < config["max_retries"] - 1:
                time.sleep(5 * (2 ** attempt))
                continue
            print(f"  ERREUR {e!r} pour la recherche {query!r}", file=sys.stderr)
            return 0

    return 0


def worker_task(job_data: tuple) -> tuple:
    dataset_key, iid, concept, config = job_data
    query = concept.replace("_", " ")
    is_in_snomed = check_concept_in_snomed(query, config)
    return dataset_key, iid, concept, is_in_snomed


def extract_jobs_from_nested_json(data: dict, config: dict) -> list:
    jobs = []
    if isinstance(data, dict):
        for dataset_key, items in data.items():
            if isinstance(items, list):
                for it in items:
                    iid = it.get("item_id", "")
                    concept = it.get("concept_proposed", "").strip() if it.get("concept_proposed") else ""
                    if concept:
                        jobs.append((dataset_key, iid, concept, config))
    elif isinstance(data, list):
        for it in data:
            iid = it.get("item_id", "")
            concept = it.get("concept_proposed", "").strip() if it.get("concept_proposed") else ""
            if concept:
                jobs.append(("default", iid, concept, config))
    return jobs


def get_children_fhir(parent_id: str, direct_only: bool = True, limit: int = 50) -> list[dict[str, str]]:
    url = "https://snowstorm.snomedtools.org/fhir/ValueSet/$expand"
    operator = "<" if direct_only else "<<"
    ecl = f"{operator} {parent_id}"
    
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
    if parent_id not in _CHILDREN_CACHE:
        _CHILDREN_CACHE[parent_id] = get_children_fhir(parent_id)
    return _CHILDREN_CACHE[parent_id]


def compute_embedding(concept: str | dict[str, str]) -> np.ndarray:
    tokenizer, encoder = _get_model()

    text = concept if isinstance(concept, str) else concept["display"]
    text = text.replace("_", " ")
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = encoder(**inputs)

    cls_rep = outputs.last_hidden_state[:, 0, :]
    normalized_tensor = F.normalize(cls_rep, p=2, dim=1)

    return normalized_tensor.squeeze(0).cpu().numpy()


def compute_embeddings(list_concept: list[dict[str, str]]) -> np.ndarray:
    if not list_concept:
        return np.empty((0, 768))
    return np.array([compute_embedding(concept) for concept in list_concept])


def give_index_more_similar(embedding: np.ndarray, list_embeddings: np.ndarray) -> tuple[float, int]:
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    similarity_scores = (embedding @ list_embeddings.T)[0]
    index = int(np.argmax(similarity_scores))
    score = float(similarity_scores[index])
    return score, index


def search_similar_concepts(
    concept: str | dict[str, str],
    list_categorys: list[dict[str, str]],
    categorys_embeddings: np.ndarray | None = None,
    threshold: float = 0.90,
) -> bool:
    concept_embedding = compute_embedding(concept)
    if categorys_embeddings is None:
        categorys_embeddings = compute_embeddings(list_categorys)

    top_score, index = give_index_more_similar(concept_embedding, categorys_embeddings)
    current_best_concept = list_categorys[index]

    while True:
        childs = get_children_cached(current_best_concept["code"])

        if not childs:
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
    if df.empty:
        return df

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
    parser = argparse.ArgumentParser(description="Biomedical Information Extraction Pipeline")
    parser.add_argument("--data", nargs="*", help="Path to zero or more input files (JSON)")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["api_key"] = get_api_key()

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

    with Pool(processes=config["num_workers"]) as pool:
        results = pool.map(worker_task, all_jobs)

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

    df_not_in_snomed = df[df["Is in SNOMED"] == 0]
    df_filter_not_in_snomed = filter_non_found_snomed_df(df_not_in_snomed)
    n_in_snomed_with_filter = n_in_snomed + len(df_filter_not_in_snomed)

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