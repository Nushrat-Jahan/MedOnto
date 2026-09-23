# MedOnto

**MedOnto** is a project on **LLM-assisted ontology engineering from heterogeneous health data**.

The project explores how different forms of health data can be semantically interpreted, aligned with existing biomedical terminologies, and transformed into ontology-oriented representations.

## Project Tracks

MedOnto currently investigates three types of health data:

### Structured Cohort Data
Semantic interpretation of structured clinical variables, ontology alignment, and LLM-assisted mapping from tabular data to RDF.

### Questionnaire and Assessment Data
Semantic interpretation of questionnaire items, alignment of measured clinical or psychological concepts with biomedical ontologies, and construction of questionnaire-oriented semantic models.

### Medical and Mental-Health Text
Extraction of medical or psychological concepts and relations from unstructured text, followed by ontology alignment and construction of small ontology fragments.

## General Workflow

The exact workflow depends on the type of source data, but the common process is:

**Source data → LLM-assisted semantic interpretation → ontology alignment → semantic representation → validation**


## Common Technical Setup

- **LLM:** Qwen2.5-7B-Instruct
- **Primary reference terminology:** SNOMED CT
- **Semantic technologies:** RDF, OWL
- **Mapping technology:** RML where applicable
- **Validation:** Correct / Incorrect / Uncertain

LLM-generated ontology concepts or identifiers are not accepted automatically and must be verified.

When no appropriate existing ontology concept is found, the concept may be recorded as a **Candidate Project Concept**.

## Repository Structure

Development is organized using separate branches. The `main` branch contains the shared project documentation and reviewed work.

Each branch is used for the corresponding work and may contain:

- notebooks and scripts;
- prompts;
- mapping files;
- ontology/RDF outputs;
- validation results;
- experimental documentation.

## Reproducibility

Experiments should preserve, when applicable:

- the original input or source reference;
- the exact prompt used;
- the raw LLM output;
- ontology mappings;
- validated results;
- corrections and uncertainty notes.

The objective is to make each transformation from source data to ontology content **traceable and reproducible**.

## Data

External datasets are not automatically redistributed through this repository.

Each dataset remains subject to its original license, access conditions, and terms of use.

## License

Original code and project materials developed in this repository are licensed under the **Apache License 2.0**.

External datasets, models, terminologies, ontologies, questionnaires, and other third-party resources remain subject to their respective licenses and terms of use.
