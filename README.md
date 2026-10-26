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

## The `architecture-commun` branch

This branch is the **shared foundation** of the project. Each student merges it into their own branch
and reorganizes their code on the same structure, so that the three tracks can be merged together
without conflicts.

```text
MedOnto/
├── README.md              ← project documentation (this file)
├── LICENSE
└── src/
    ├── __init__.py
    ├── postprocess.py     ← shared entry point to standardize model outputs
    ├── verification.py    ← shared logic to accept or reject model outputs
    ├── utils.py           ← shared helper functions (file I/O, parsing)
    ├── text/              ← track: medical and mental-health text
    │   ├── __init__.py
    │   ├── pipeline.py
    │   ├── preprocess.py
    │   └── spanextraction.py
    ├── bdd/               ← track: structured cohort data
    │   ├── __init__.py
    │   ├── task1.py
    │   ├── extract_variables.py
    │   └── build_ontology.py
    └── questions/         ← track: questionnaire and assessment data
        ├── __init__.py
        ├── extraction_qwen_automatisee.py
        ├── recherche_snomed_B2.py
        ├── decision_match_C2.py
        ├── generation_ttl_D2.py
        └── valider_ttl_D2.py
```

### Shared modules (`src/`)

- **`postprocess.py`** is the **entry point to standardize our data**. Every track passes the raw
  outputs of the model through it, so that all results share the same format before verification:
  text normalization (lowercase, whitespace) and removal of meaningless items (punctuation,
  stopwords, anonymized tokens, digits only, blacklisted terms, lab-value patterns).
- **`verification.py`** implements the **shared SNOMED CT verification pipeline**. Each extracted
  concept is checked with a three-step fallback sequence and receives a structured result
  (`concept`, `in_snomed`, `verification_step`). The results are aggregated into a pandas DataFrame.
- **`utils.py`** gathers the **shared helper functions** used by every track: reading text, PDF,
  TOML and JSON files, writing JSON files, parsing concept and triplet lists from raw model output,
  and the common `Triplet` structure (subject, relation, object).

### Verification pipeline (`src/verification.py`)

For each concept produced by the extraction stage, the worker applies the following fallback
sequence:

1. **Direct SNOMED lookup.** The proposed concept is queried against SNOMED CT through the
   BioPortal API. If a real SNOMED identifier is returned, verification stops at **step 1**.
2. **Synonym lookup and child refinement.** If step 1 fails, the `synonyms` list emitted by the
   span-extraction task is checked one by one. When a synonym matches, the pipeline retrieves the
   children of the matched SNOMED concept and compares them with the original proposed concept
   using the clinical embedding model. It descends the hierarchy as long as a child is more similar
   to the original concept, then keeps the best child; if no child is closer, the matched synonym is
   kept. This remains **step 2**.
3. **Embedding fallback.** If no synonym matches, the original concept is passed to the custom
   hierarchical embedding search over SNOMED. The closest concept is accepted when its cosine
   similarity reaches `embedding_threshold` (default `0.90`). This is **step 3**.

If no match is found at any step, the original proposed concept is returned with `in_snomed = 0`.
`verification_step` is then `3`, because that is where the final negative decision is made.

Each worker returns a JSON-serializable record:

| Field | Description |
|---|---|
| `concept` | Final matching concept name, or the original proposed concept if no match was found |
| `in_snomed` | `1` if a match was found at any step, otherwise `0` |
| `verification_step` | Step that determined the result: `1`, `2`, or `3` |

The worker results are aggregated into a pandas DataFrame with the columns:

| Column | Description |
|---|---|
| `Source Dataset` | Source JSON key or dataset name |
| `Item ID` | Extracted item identifier |
| `Concept (A2)` | Original proposed concept |
| `Matched Concept` | Concept retained after verification |
| `Is in SNOMED` | `1` if matched at any step, otherwise `0` |
| `Verification Step` | `1`, `2`, or `3` |

The pipeline requires a `BIOPORTAL_API_KEY` in the `.env` file and can be run from the project root:

```bash
python -m src.verification --data res/test.json
```

The final output is the verification DataFrame.

### Track packages

Each track has its own package inside `src/`:

| Package | Track |
|---|---|
| [`src/text/`](src/text/README.md) | Medical and mental-health text |
| `src/bdd/` | Structured cohort data |
| `src/questions/` | Questionnaire and assessment data |

Every folder of `src/` is a Python package (`__init__.py`). Code is run and imported from the
project root, for example `from src.utils import load_json_file` or
`from src.text.preprocess import preprocess`.

The [`src/text/` README](src/text/README.md) documents the text-extraction pipeline in detail:
how to configure the models and prompts, and how to run `src.text.pipeline` from the command
line or as a Python function.

## Common technical setup

- **LLM:** Qwen2.5-7B-Instruct
- **Reference terminology:** SNOMED CT only. When no appropriate concept exists, the concept is
  recorded as a **Candidate Project Concept**.
- **Semantic technologies:** RDF, OWL
- **Validation:** automatic SNOMED CT verification (`in_snomed`, `verification_step`), followed by
  manual review status Correct / Incorrect / Uncertain

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

