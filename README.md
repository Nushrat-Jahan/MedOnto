# MedOnto

**MedOnto** is a project on **LLM-assisted ontology engineering from heterogeneous health data**.

The project studies how health data can be semantically interpreted by a large language model
(Qwen2.5), aligned with a reference terminology (SNOMED CT) and transformed into ontology content
(concepts, relations, RDF/OWL fragments), and how data preparation affects the quality of the result.

## Project tracks

| Track | Source data | Goal |
|---|---|---|
| Structured cohort data | Tabular variables (CSV) | Turn variables into normalized concepts, categories and relations, then into an ontology fragment |
| Questionnaire and assessment data | Questionnaire items | Align measured clinical or psychological concepts with biomedical ontologies |
| Medical and mental-health text | Unstructured documents | Extract concepts and relation triplets, then align and build ontology fragments |

Common workflow:

**Source data → LLM-assisted semantic interpretation → post-processing → verification → ontology alignment → semantic representation**

## Repository structure

The project is organized in **two parts**: an **extraction** specific to each type of data, which
always produces the same JSON (the common format), then a **verification** shared by all tracks.

```text
extraction (one per track) → common JSON → shared verification (SNOMED CT)
```

```text
MedOnto/
├── README.md                    ← project documentation (this file)
├── LICENSE
├── configs/
│   ├── models/                  ← model settings (text track)
│   ├── systems_prompt/          ← extraction prompts: shared system prompt, one user template per track
│   └── verification_prompts/    ← prompts given to Qwen during the verification
└── src/
    ├── __init__.py
    ├── postprocess.py           ← shared cleaning of the model outputs (all tracks)
    ├── utils.py                 ← shared helper functions (file I/O, parsing)
    │
    ├── bdd/                     ← EXTRACTION · structured cohort data
    │   ├── pipeline.py          ← entry point
    │   ├── extract_variables.py ← step 1: variables of the dataset
    │   └── extract_concepts.py  ← step 2: concepts proposed by Qwen, in the common JSON format
    ├── text/                    ← EXTRACTION · medical and mental-health text
    │   ├── pipeline.py          ← entry point
    │   ├── preprocess.py
    │   └── spanextraction.py
    ├── questions/               ← EXTRACTION · questionnaire and assessment data
    │   ├── extraction_questions.py
    │   └── items.json
    │
    └── verification/            ← SHARED VERIFICATION · common JSON → SNOMED CT
        ├── snomed_api.py        ← questions to the BioPortal API (exact search, broad search, children)
        ├── llm_choice.py        ← questions to Qwen (choose a candidate, propose another name)
        ├── snomed_check.py      ← verification steps
        └── voir_api.py          ← small tool showing the raw answer of the API for a term
```

Not versioned: `data/` (datasets), `outputs/` (results, confidential) and `.env` (API key).

Every folder of `src/` is a Python package (`__init__.py`). Code is run and imported from the
project root, for example `from src.utils import load_json_file`.

### Shared modules

- **`postprocess.py`** standardizes the raw outputs of the model, so that all tracks share the same
  format before verification: text normalization (lowercase, whitespace), removal of meaningless
  items, and removal of near-duplicate concepts.
- **`utils.py`** gathers the shared helper functions: reading text, PDF, TOML and JSON files,
  writing JSON files, parsing concept and triplet lists from raw model output.

### The shared verification (`src/verification/`)

For each concept of the common JSON, the verification stops as soon as one step finds it:

1. the concept itself is searched in SNOMED CT (exact search);
2. otherwise, its synonyms (exact search);
3. otherwise, the API proposes candidates (broad search) and Qwen chooses the closest one, or
   proposes another name; the direct children of the chosen concept are then examined;
4. otherwise, embedding similarity;
5. otherwise, the concept becomes a candidate proposed to the experts.

Qwen only chooses among real SNOMED CT concepts returned by the API: it cannot invent an identifier.
Concepts found with Qwen are marked "to confirm" for the experts.

The [`src/text/` README](src/text/README.md) documents the text-extraction pipeline in detail.

## How to run

From the project root, with Python 3.11 and Ollama running (model `qwen2.5:7b`):

```text
# Extraction, structured data track: dataset → common JSON
python -m src.bdd.pipeline "data/<dataset>.csv"
python -m src.bdd.pipeline "data/<dataset>.xlsx" --label-row   (if row 2 contains the labels)
→ outputs/<dataset>_<id>/concepts_<id>.json

# Raw answer of the BioPortal API for a term
python src/verification/voir_api.py "white blood cell count"
```

The verification needs a BioPortal API key in a `.env` file at the project root:
`BIOPORTAL_API_KEY=<your key>`.

## Common technical setup

- **LLM:** Qwen2.5-7B-Instruct
- **Reference terminology:** SNOMED CT only. When no appropriate concept exists, the concept is
  recorded as a **Candidate Project Concept**.
- **Semantic technologies:** RDF, OWL
- **Validation:** Correct / Incorrect / Uncertain

LLM-generated concepts and identifiers are never accepted automatically: they must be verified.

## Reproducibility

Experiments should preserve, when applicable: the source data reference, the exact prompt, the raw
LLM output, the ontology mappings, the validated results, and the corrections and uncertainty notes.

## Data

External datasets are not redistributed through this repository. Each dataset remains subject to
its original license, access conditions and terms of use.

## License

Original code and project materials are licensed under the **Apache License 2.0**. External
datasets, models, terminologies, ontologies and questionnaires remain subject to their own licenses
and terms of use.

