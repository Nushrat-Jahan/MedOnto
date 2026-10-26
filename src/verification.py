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
    "num_workers": cpu_count(),
    "embedding_threshold": 0.90,
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
_SNOMED_CATEGORY_EMBEDDINGS: np.ndarray | None = None


def get_api_key() -> str:
    key = os.environ.get("BIOPORTAL_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "BIOPORTAL_API_KEY manquante dans le fichier .env.\n"
            "Veuillez ajouter la ligne suivante dans votre .env :\n"
            "BIOPORTAL_API_KEY=votre_cle_ici"
        )
    return key


def search_snomed_concept(query: str, config: dict) -> dict[str, str] | None:
    """Return the first real SNOMED concept matching ``query``, or ``None``."""
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
                    label = str(concept.get("prefLabel") or query).strip() or query
                    return {"id": snomed_id, "label": label}
            return None

        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < config["max_retries"] - 1:
                time.sleep(5 * (2 ** attempt))
                continue
            print(f"  ERREUR HTTP {e.code} pour la recherche {query!r}", file=sys.stderr)
            return None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt < config["max_retries"] - 1:
                time.sleep(5 * (2 ** attempt))
                continue
            print(f"  ERREUR {e!r} pour la recherche {query!r}", file=sys.stderr)
            return None

    return None


def check_concept_in_snomed(query: str, config: dict) -> int:
    """Backward-compatible 0/1 wrapper around :func:`search_snomed_concept`."""
    return 1 if search_snomed_concept(query, config) else 0


def normalize_synonyms(raw_synonyms) -> list[str]:
    """Normalize the ``synonyms`` field emitted by the span-extraction task."""
    if raw_synonyms is None:
        return []
    if isinstance(raw_synonyms, str):
        raw_synonyms = raw_synonyms.replace(";", ",").split(",")

    synonyms: list[str] = []
    for raw in raw_synonyms:
        synonym = str(raw).strip().replace("_", " ")
        synonym = " ".join(synonym.split())
        if synonym and synonym.lower() not in {s.lower() for s in synonyms}:
            synonyms.append(synonym)
    return synonyms


def refine_synonym_match(
    original_concept: str,
    synonym: str,
    snomed_match: dict[str, str],
) -> str:
    """Search the matched SNOMED concept's children for a closer match.

    The child hierarchy is explored as long as a child is more similar to the
    original concept than the synonym/preferred label that was initially found.
    Returns the original synonym when no more relevant child exists.
    """
    concept_embedding = compute_embedding(original_concept)

    baseline_texts = {synonym, snomed_match.get("label") or synonym}
    baseline_score = max(
        float(concept_embedding @ compute_embedding(text))
        for text in baseline_texts
        if text
    )

    best_child: dict[str, str] | None = None
    best_score = baseline_score
    parent = snomed_match

    while True:
        children = get_children_cached(parent["id"])
        if not children:
            break

        children_embeddings = compute_embeddings(children)
        child_score, child_index = give_index_more_similar(
            concept_embedding, children_embeddings
        )

        if child_score <= best_score:
            break

        best_child = children[child_index]
        best_score = child_score
        parent = {"id": best_child["code"], "display": best_child["display"]}

    return best_child["display"] if best_child else synonym


def verify_concept(concept: str, synonyms: list[str], config: dict) -> dict:
    """Verify a concept against SNOMED with the three-step fallback sequence.

    Returns a JSON-serializable dict with exactly:
      - ``concept``: final matching concept name, or the original concept on failure;
      - ``in_snomed``: 1 if a match was found at any step, otherwise 0;
      - ``verification_step``: 1, 2 or 3, the step that determined the result.
    """
    concept = (concept or "").strip()
    query = concept.replace("_", " ")

    # Step 1 — direct check of the proposed concept.
    if check_concept_in_snomed(query, config):
        return {
            "concept": concept,
            "in_snomed": 1,
            "verification_step": 1,
        }

    # Step 2 — check the synonyms produced by the span-extraction task. When a
    # synonym matches, look for a more specific child that is closer to the
    # original concept.
    for synonym in normalize_synonyms(synonyms):
        if synonym.lower() == query.lower():
            continue
        match = search_snomed_concept(synonym.replace("_", " "), config)
        if match:
            final_concept = refine_synonym_match(query, synonym, match)
            return {
                "concept": final_concept,
                "in_snomed": 1,
                "verification_step": 2,
            }

    # Step 3 — custom embedding fallback: closest SNOMED concept.
    closest, _score = find_closest_snomed_concept(
        query,
        threshold=config.get("embedding_threshold", DEFAULT_CONFIG["embedding_threshold"]),
    )
    if closest is not None:
        return {
            "concept": closest["display"],
            "in_snomed": 1,
            "verification_step": 3,
        }

    return {
        "concept": concept,
        "in_snomed": 0,
        "verification_step": 3,
    }


def worker_task(job_data: tuple) -> dict:
    dataset_key, iid, concept, synonyms, config = job_data
    result = verify_concept(concept, synonyms, config)
    return {
        "dataset_key": dataset_key,
        "item_id": iid,
        "proposed_concept": concept,
        **result,
    }


def _extract_jobs_from_items(dataset_key: str, items: list, config: dict) -> list:
    jobs = []
    for it in items:
        if not isinstance(it, dict):
            continue
        concept = str(
            it.get("concept_proposed") or it.get("proposed_concept") or ""
        ).strip()
        if not concept:
            continue
        jobs.append((
            dataset_key,
            it.get("item_id", ""),
            concept,
            normalize_synonyms(it.get("synonyms")),
            config,
        ))
    return jobs


def extract_jobs_from_nested_json(data: dict, config: dict) -> list:
    jobs = []
    if isinstance(data, dict):
        for dataset_key, items in data.items():
            if isinstance(items, list):
                jobs.extend(_extract_jobs_from_items(dataset_key, items, config))
    elif isinstance(data, list):
        jobs.extend(_extract_jobs_from_items("default", data, config))
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


def get_snomed_category_embeddings() -> np.ndarray:
    """Compute the top-level SNOMED embeddings once per worker process."""
    global _SNOMED_CATEGORY_EMBEDDINGS
    if _SNOMED_CATEGORY_EMBEDDINGS is None:
        _SNOMED_CATEGORY_EMBEDDINGS = compute_embeddings(SNOMED_TOP_LEVEL_CATEGORY)
    return _SNOMED_CATEGORY_EMBEDDINGS


def find_closest_snomed_concept(
    concept: str | dict[str, str],
    list_categorys: list[dict[str, str]] | None = None,
    categorys_embeddings: np.ndarray | None = None,
    threshold: float = 0.90,
) -> tuple[dict[str, str] | None, float]:
    """Run the custom hierarchical embedding search.

    Returns ``(closest_concept, score)`` when the best cosine similarity reaches
    ``threshold``; otherwise ``(None, score)``.
    """
    if list_categorys is None:
        list_categorys = SNOMED_TOP_LEVEL_CATEGORY
    if categorys_embeddings is None:
        categorys_embeddings = get_snomed_category_embeddings()

    concept_embedding = compute_embedding(concept)
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

    if top_score >= threshold:
        return current_best_concept, top_score
    return None, top_score


def search_similar_concepts(
    concept: str | dict[str, str],
    list_categorys: list[dict[str, str]],
    categorys_embeddings: np.ndarray | None = None,
    threshold: float = 0.90,
) -> bool:
    closest, _score = find_closest_snomed_concept(
        concept,
        list_categorys,
        categorys_embeddings,
        threshold,
    )
    return closest is not None


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


def main() -> pd.DataFrame:
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
        return pd.DataFrame(columns=[
            "Source Dataset",
            "Item ID",
            "Concept (A2)",
            "Matched Concept",
            "Is in SNOMED",
            "Verification Step",
        ])

    with Pool(processes=config["num_workers"]) as pool:
        results = pool.map(worker_task, all_jobs)

    rows = []
    for result in results:
        rows.append({
            "Source Dataset": result["dataset_key"],
            "Item ID": result["item_id"],
            "Concept (A2)": result["proposed_concept"],
            "Matched Concept": result["concept"],
            "Is in SNOMED": result["in_snomed"],
            "Verification Step": result["verification_step"],
        })

    df = pd.DataFrame(rows)
    n = len(df)
    n_in_snomed = int((df["Is in SNOMED"] == 1).sum())
    by_step = df["Verification Step"].value_counts().sort_index()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print(f"\nTotal: {n} | Trouvés dans SNOMED: {n_in_snomed}")
    print("Résultats par étape de vérification : "
          + ", ".join(f"étape {step}: {count}" for step, count in by_step.items()))

    # Final output: the verification DataFrame.
    print(df)
    return df


if __name__ == "__main__":
    main()