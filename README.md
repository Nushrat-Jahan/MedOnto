# MedOnto
LLM-assisted ontology construction from heterogeneous health data.
# Mapping de questionnaires de santé mentale vers SNOMED CT

**Pipeline hybride LLM + BioPortal**
*Qwen2.5-7B-Instruct + API BioPortal*

# English Version

## Objective

This project aims to link items from mental health screening questionnaires (**PHQ-9**, **GAD-7**, **Epworth Sleepiness Scale**) to standardized clinical concepts in **SNOMED CT**.

Two approaches are compared:

* manual validation through a search in BioPortal ;
* an automated pipeline using a local LLM (**Qwen2.5-7B-Instruct**), the BioPortal API, and automatic generation of RDF/Turtle fragments.

The ultimate goal is to produce a **lightweight ontology** connecting each questionnaire item to a validated clinical concept.

---

## Items Processed

The project currently covers **9 items from 3 questionnaires**:

| Questionnaire                | Items                          |
| ---------------------------- | ------------------------------ |
| **PHQ-9**                    | `phq2`, `phq4`, `phq7`, `phq9` |
| **GAD-7**                    | `gad1`, `gad4`, `gad6`          |
| **Epworth Sleepiness Scale** | `epw4`, `epw8`                 |

---

## Repository Structure

```text
Automatisation/
├── src/
│   ├── extraction_qwen_automatisee.py     # A2: clinical concept extraction using Qwen
│   ├── recherche_snomed_B2.py             # B2: SNOMED candidate search via BioPortal
│   ├── decision_match_C2.py               # C2: match decision (Exact/Broader/Narrower/No Match)
│   ├── generation_ttl_D2.py               # D2: RDF/Turtle fragment generation
│   ├── valider_ttl_D2.py                  # Syntax and semantic validation of the TTL
│   └── baseline_top1_C2.py                # baseline: BioPortal rank 1, without C2 selection
├── result/
│   ├── resultats_A2.csv / .json           # A2 outputs
│   ├── resultats_B2_candidats.csv         # SNOMED candidates (B2)
│   ├── resultats_C2_match.csv             # Match decisions (C2)
│   ├── resultats_baseline_top1.csv        # Baseline results
│   ├── comparaison_manuel_vs_automatique.csv
│   └── questionnaire_ontology_AUTO_D2.ttl # Automatically generated RDF fragment
├── Documentation/                         # Documentation (PDF)
└── .env                                   # BioPortal API key (not versioned)
```

---

## Automated Pipeline

The pipeline follows these steps:

```text
Items
  │
  ▼
A2 — Qwen
Clinical concept extraction
  │
  ▼
B2 — BioPortal
Search for real SNOMED CT candidates
  │
  ▼
C2 — Qwen
Match decision
(Exact / Broader / Narrower / No Match)
  │
  ▼
D2 — RDF/Turtle generation
  │
  ▼
Validation via rdflib
```

### Core Principle

> **The LLM never directly manipulates SNOMED CT.**

Only the **BioPortal API** provides real SNOMED CT identifiers.

Any attempt to invent an identifier is automatically rejected.

---

## Configuration

### 1. LLM Model

The pipeline uses **Ollama** with the model:

```text
qwen2.5:7b-instruct
```

### 2. Python Dependencies

Install the dependencies using:

```bash
pip install ollama python-dotenv rdflib
```

### 3. BioPortal API Key

Create a `.env` file in the `Automatisation/` directory:

```env
BIOPORTAL_API_KEY=your_api_key
```

The `.env` file contains the BioPortal API key and must not be versioned.

---

## Execution

The scripts must be run in the following order:

### Step A2 — Clinical Concept Extraction

```bash
python extraction_qwen_automatisee.py
```

### Step B2 — SNOMED CT Candidate Search

```bash
python recherche_snomed_B2.py
```

### Step C2 — Match Decision

```bash
python decision_match_C2.py
```

### Step D2 — RDF/Turtle Fragment Generation

```bash
python generation_ttl_D2.py
```

### TTL Validation

```bash
python valider_ttl_D2.py
```

---

## Current Results

The comparison between manual validation and the automated pipeline currently covers **9 items**.

| Result                                                           | Number |
| ---------------------------------------------------------------- | -----: |
| Total agreements                                                 |      3 |
| Partial agreements *(same concept, different match)*             |      2 |
| Disagreements *(different concept, including one wrongly inferred diagnosis)* | 2 |
| False negatives *(correct concept present but not selected)*     |      2 |

---

## Observed Limitations

Several limitations have been observed during testing:

* The **7B** model does not always follow the prompt instructions, particularly for the **Epworth** items.
* An imprecise concept in **A2** may prevent BioPortal from returning the correct candidate.
* The LLM returns "no match" if it does not find the same thing as the Qwen concept, even though very similar and more specific terms are candidates.

# Version en français 

## Objectif

Ce projet vise à relier des items de questionnaires de dépistage en santé mentale (**PHQ-9**, **GAD-7**, **Epworth Sleepiness Scale**) à des concepts cliniques standardisés **SNOMED CT**.

Deux approches sont comparées :

* une validation manuelle via une recherche dans BioPortal ;
* un pipeline automatisé utilisant un LLM local (**Qwen2.5-7B-Instruct**), l'API BioPortal et une génération automatique de fragments RDF/Turtle.

L'objectif final est de produire une **ontologie légère** connectant chaque item de questionnaire à un concept clinique validé.

---

## Items traités

Le projet porte actuellement sur **9 items issus de 3 questionnaires** :

| Questionnaire                | Items                          |
| ---------------------------- | ------------------------------ |
| **PHQ-9**                    | `phq2`, `phq4`, `phq7`, `phq9` |
| **GAD-7**                    | `gad1`, `gad4`, `gad6`         |
| **Epworth Sleepiness Scale** | `epw4`, `epw8`                 |

---

## Structure du dépôt

```text
Automatisation/
├── src/
│   ├── extraction_qwen_automatisee.py     # A2 : extraction du concept clinique via Qwen
│   ├── recherche_snomed_B2.py             # B2 : recherche de candidats SNOMED via BioPortal
│   ├── decision_match_C2.py               # C2 : décision du match (Exact/Broader/Narrower/No Match)
│   ├── generation_ttl_D2.py               # D2 : génération du fragment RDF/Turtle
│   ├── valider_ttl_D2.py                  # validation syntaxique et sémantique du TTL
│   └── baseline_top1_C2.py                # baseline : rang 1 de BioPortal, sans le choix de C2
├── result/
│   ├── resultats_A2.csv / .json           # sorties de A2
│   ├── resultats_B2_candidats.csv         # candidats SNOMED (B2)
│   ├── resultats_C2_match.csv             # décisions de match (C2)
│   ├── resultats_baseline_top1.csv        # résultats de la baseline
│   ├── comparaison_manuel_vs_automatique.csv
│   └── questionnaire_ontology_AUTO_D2.ttl # fragment RDF généré automatiquement
├── Documentation/                         # documentation (PDF)
└── .env                                   # clé API BioPortal (non versionné)
```

---

## Pipeline automatisé

Le pipeline suit les étapes suivantes :

```text
Items
  │
  ▼
A2 — Qwen
Extraction du concept clinique
  │
  ▼
B2 — BioPortal
Recherche de candidats SNOMED CT réels
  │
  ▼
C2 — Qwen
Décision du match
(Exact / Broader / Narrower / No Match)
  │
  ▼
D2 — Génération RDF/Turtle
  │
  ▼
Validation via rdflib
```

### Principe central

> **Le LLM ne manipule jamais directement SNOMED CT.**

Seule l'**API BioPortal** fournit les identifiants SNOMED CT réels.

Toute tentative d'inventer un identifiant est automatiquement rejetée.

---

## Configuration

### 1. Modèle LLM

Le pipeline utilise **Ollama** avec le modèle :

```text
qwen2.5:7b-instruct
```

### 2. Dépendances Python

Installer les dépendances avec :

```bash
pip install ollama python-dotenv rdflib
```

### 3. Clé API BioPortal

Créer un fichier `.env` dans le dossier `Automatisation/` :

```env
BIOPORTAL_API_KEY=ta_clé_api
```

Le fichier `.env` contient la clé API BioPortal et ne doit pas être versionné.

---

## Exécution

Les scripts doivent être lancés dans l'ordre suivant :

### Étape A2 — Extraction du concept clinique

```bash
python extraction_qwen_automatisee.py
```

### Étape B2 — Recherche des candidats SNOMED CT

```bash
python recherche_snomed_B2.py
```

### Étape C2 — Décision du match

```bash
python decision_match_C2.py
```

### Étape D2 — Génération du fragment RDF/Turtle

```bash
python generation_ttl_D2.py
```

### Validation du TTL

```bash
python valider_ttl_D2.py
```

---

## Résultats actuels

La comparaison entre la validation manuelle et le pipeline automatique porte actuellement sur **9 items**.

| Résultat                                                           | Nombre |
| ------------------------------------------------------------------ | -----: |
| Accords totaux                                                     |      3 |
| Accords partiels *(même concept, match différent)*                 |      2 |
| Désaccords *(concept différent, dont un diagnostic inféré à tort)* |      2 |
| Faux négatifs *(bon concept présent mais non sélectionné)*         |      2 |

---

## Limites observées

Plusieurs limites ont été observées lors des tests :

* Le modèle **7B** ne suit pas toujours les consignes du prompt, notamment pour les items **Epworth**.
* Un concept imprécis en **A2** peut empêcher BioPortal de retourner le bon candidat.
*Le LLM, met "no match", s'il ne trouve pas la même chose que le concept de QWEN sachant que des mots très proche et plus spécifique sont des candidats

---

## À faire

*  Testez une méthode de base sans LLM (prendre directement le premier résultat renvoyé par BioPortal) afin de mesurer ce que le LLM apporte réellement.
*  Extension à un **jeu d'items plus large**.
*  J'essaye de chercher tous les filles, et je demande à Qwen de choisir entre les filles et ce qu'il m'a proposé en premier
  
