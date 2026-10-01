"""
Étape B2 du pipeline automatique — Recherche de candidats SNOMED CT RÉELS.

Pour chaque concept proposé par A2 (resultats_A2.json), interroge l'API
BioPortal (https://data.bioontology.org/search, ontologie SNOMEDCT) et
conserve 3 à 5 candidats par item :
  - ID SNOMED (extrait de l'URI "@id" du concept dans la réponse API),
  - label préféré (prefLabel),
  - définition (definition, si présente),
  - 3 premiers synonymes (synonym).

AUCUN LLM n'intervient à cette étape : toute valeur SNOMED (ID, label,
définition, synonyme) provient directement de la réponse de l'API BioPortal.


Clé API : variable d'environnement BIOPORTAL_API_KEY, jamais écrite dans
le code ni dans les sorties.

Sortie : resultats_B2_candidats.csv (une ligne par candidat, 3-5 par item).
"""

import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
load_dotenv()

API_URL = "https://data.bioontology.org/search"
ONTOLOGY = "SNOMEDCT"
PAGESIZE = 10
MAX_WORKERS = 3
TIMEOUT = 30
MAX_RETRIES = 3
INPUT_JSON = "resultats_A2.json"
OUTPUT_CSV = "resultats_B2_candidats.csv"



def get_api_key() -> str:
    key = os.environ.get("BIOPORTAL_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "BIOPORTAL_API_KEY manquante. Deux options :\n"
            "  1) Créer un fichier .env dans ce dossier avec la ligne :\n"
            "     BIOPORTAL_API_KEY=ta_cle_ici\n"
            "  2) Ou définir la variable dans ce terminal avant de relancer B2 :\n"
            "     PowerShell : $env:BIOPORTAL_API_KEY=\"ta_cle_ici\""
        )
    return key


def search_once(query: str, key: str) -> list:
    """Une requête BioPortal ; candidats copiés tels quels de la réponse API."""
    params = urllib.parse.urlencode({
        "q": query,
        "ontologies": ONTOLOGY,
        "apikey": key,
        "pagesize": PAGESIZE,
        "include": "prefLabel,synonym,definition,notation",
    })
    req = urllib.request.Request(
        f"{API_URL}?{params}",
        headers={"User-Agent": "pipeline-LIRMM-A2D2/1.0"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        data = json.load(resp)

    candidates, seen = [], set()
    for concept in data.get("collection", []):
        snomed_id = str(concept.get("@id", "")).rsplit("/", 1)[-1]
        if not snomed_id.isdigit() or snomed_id in seen:
            continue  # on ne garde que de vrais IDs SNOMED issus de la réponse API
        seen.add(snomed_id)
        definitions = concept.get("definition") or []
        candidates.append({
            "snomed_id": snomed_id,
            "label": str(concept.get("prefLabel", "")).strip(),
            "definition": str(definitions[0]).strip() if definitions else "",
            "synonyms": "; ".join(str(s) for s in (concept.get("synonym") or [])[:3]),
        })
    return candidates[:PAGESIZE]


def search_candidates(query: str, key: str) -> list:
    """Requêtes avec retries sur les erreurs transitoires (429, 5xx, réseau)."""
    for attempt in range(MAX_RETRIES):
        try:
            return search_once(query, key)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < MAX_RETRIES - 1:
                time.sleep(5 * (2 ** attempt))
                continue
            print(f"  ERREUR HTTP {e.code} pour la recherche {query!r}")
            return []
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(5 * (2 ** attempt))
                continue
            print(f"  ERREUR {e!r} pour la recherche {query!r}")
            return []
    return []


def main() -> None:
    key = get_api_key()

    with open(INPUT_JSON, encoding="utf-8") as f:
        items = json.load(f)

    jobs = [(it["item_id"], it.get("concept_proposed", "").strip())
            for it in items if it.get("concept_proposed", "").strip()]

    def work(job):
        iid, concept = job
        query = concept.replace("_", " ")
        cands = search_candidates(query, key)
        print(f"→ {iid} : {concept!r} -> {len(cands)} candidat(s) BioPortal (requête : {query!r})")
        return iid, concept, cands

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(work, jobs))

    rows = []
    for iid, concept, cands in results:
        for rank, c in enumerate(cands, 1):
            rows.append({
                "Item ID": iid,
                "Concept (A2)": concept,
                "Rang": rank,
                "SNOMED ID": c["snomed_id"],
                "Preferred label": c["label"],
                "Definition": c["definition"],
                "Synonyms": c["synonyms"],
            })

    fieldnames = ["Item ID", "Concept (A2)", "Rang", "SNOMED ID",
                  "Preferred label", "Definition", "Synonyms"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nTerminé. {len(rows)} candidat(s) écrits dans {OUTPUT_CSV}")


if __name__ == "__main__":
    main()