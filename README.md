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
One command, `src/main.py`, runs the whole chain for every type of data.

```text
data/bdd/            → extraction (structured data)  ┐
data/questionnaire/  → extraction (questionnaires)   ├→ common JSON → shared verification (SNOMED CT) → Excel file for the experts
data/texte/          → extraction (texts)            ┘
```

All tracks use the **same model**: Qwen 2.5 7B, run locally with Ollama (`qwen2.5:7b`, temperature 0).

```text
MedOnto/
├── README.md                    ← project documentation (this file)
├── LICENSE
├── configs/
│   ├── models/                  ← text track: tokenizer (to truncate long texts) and GLiNER filter
│   ├── systems_prompt/          ← extraction prompts: shared system prompt, one user template per track
│   └── verification_prompts/    ← prompts given to Qwen during the verification
└── src/
    ├── main.py                  ← ENTRY POINT of the whole chain (all data, or one file)
    ├── postprocess.py           ← shared cleaning of the model outputs (all tracks)
    ├── utils.py                 ← shared helper functions (file I/O, parsing)
    │
    ├── bdd/                     ← EXTRACTION · structured cohort data (CSV, Excel)
    │   ├── extraction_bdd.py    ← run_bdd(): dataset → common JSON
    │   ├── extract_variables.py ← step 1: variables of the dataset
    │   └── extract_concepts.py  ← step 2: concepts proposed by Qwen
    ├── questions/               ← EXTRACTION · questionnaires
    │   ├── extraction_questions.py ← run_questions(): questions file → common JSON
    │   └── items.json           ← example questions file (PHQ-9, GAD-7, Epworth items)
    ├── text/                    ← EXTRACTION · medical and mental-health texts
    │   ├── extraction_text.py   ← run_text() (PDF/TXT file), run_text_hf() (Hugging Face dataset)
    │   ├── preprocess.py        ← text cleaning, GLiNER semantic filter
    │   └── spanextraction.py    ← concept generation with Qwen (Ollama)
    │
    └── verification/            ← SHARED VERIFICATION · common JSON → SNOMED CT
        ├── verification.py      ← run_verification(): common JSON → verification files
        ├── snomed_api.py        ← questions to the BioPortal API (exact search, broad search, children)
        ├── llm_choice.py        ← questions to Qwen (choose a candidate, propose another name)
        ├── snomed_check.py      ← verification steps and Excel file for the experts
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
  writing JSON files.

### The shared verification (`src/verification/`)

For each concept of the common JSON, the verification stops as soon as one step finds it:

1. the concept itself is searched in SNOMED CT (exact search);
2. otherwise, its synonyms (exact search);
3. otherwise, the API proposes candidates (broad search) and Qwen chooses the closest one, or
   proposes another name; the direct children of the chosen concept are then examined;
4. embedding similarity: **not used** — measured on our data, the current model (Bio_ClinicalBERT)
   does not separate correct concepts from wrong ones;
5. otherwise, the concept becomes a candidate proposed to the experts.

Qwen only chooses among real SNOMED CT concepts returned by the API: it cannot invent an identifier.
Concepts found with Qwen are marked "to confirm" in the Excel file for the experts.

The [`src/text/` README](src/text/README.md) documents the text-extraction pipeline in detail.

## How to run

### Requirements

- Python 3.11, with the libraries used by the project (`pandas`, `openpyxl`, `ollama`, `torch`,
  `transformers`, `pypdf`, `python-dotenv`, `tqdm`; for the text track: `datasets`, `gliner`).
- **Ollama running**, with the model `qwen2.5:7b`.
- A BioPortal API key in a `.env` file at the project root: `BIOPORTAL_API_KEY=<your key>`.

### Organize the data

Put each file in the folder of its type (the type is deduced from the folder):

```text
data/
├── bdd/              CSV or Excel datasets (an Excel file with a README sheet has its labels in row 2)
├── questionnaire/    questions files in the format of src/questions/items.json (.json)
└── texte/            clinical texts (.pdf, .txt)
                      + hf_datasets.txt: Hugging Face datasets of texts, one per line
```

`hf_datasets.txt` lists datasets read directly from Hugging Face (nothing is downloaded), one line
per dataset: its name, then the column that contains the text. Lines starting with `#` are ignored.

```text
aisc-team-a1/augmented-clinical-notes full_note
ShenLab/MentalChat16K input
```

### Run the whole chain

From the project root:

```text
python -m src.main                                     # every file of data/
python -m src.main "data/bdd/<dataset>.csv"            # one file (type deduced from its folder)
python -m src.main "elsewhere/<dataset>.csv" --type bdd   # a file stored elsewhere
python -m src.main --max-documents 3                   # texts read per Hugging Face dataset (default: 5)
```

For each file (or dataset), the chain writes in `outputs/<name>_<id>/`:

```text
concepts_<id>.json                ← extraction: concepts in the common JSON format
concepts_<id>_verification.json   ← verification: full result, all concepts
concepts_<id>_experts.xlsx        ← concepts to review by the experts (CONCEPTS and README sheets)
```

An error on one file does not stop the others; a summary is printed at the end. Each text requires
a call to Qwen (from 30 seconds to 2 minutes on a computer without GPU): keep `--max-documents` small.

### Run one step only

```text
python -m src.bdd.extraction_bdd "data/bdd/<dataset>.xlsx" --label-row      # structured data → common JSON
python -m src.questions.extraction_questions "data/questionnaire/items.json"  # questionnaire → common JSON
python -m src.text.extraction_text "data/texte/<text>.pdf" --res_path "outputs/<name>/concepts.json"
python -m src.verification.verification "outputs/<name>_<id>/concepts_<id>.json"   # common JSON → SNOMED CT
python src/verification/voir_api.py "white blood cell count"                # raw answer of the BioPortal API
```

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

