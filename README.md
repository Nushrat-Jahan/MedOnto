# MedOnto
LLM-assisted ontology construction from heterogeneous health data.
# Mapping de questionnaires de santé mentale vers SNOMED CT

**Pipeline hybride LLM + BioPortal**
*Qwen2.5-7B-Instruct + API BioPortal*

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
├── extraction_qwen_automatisee.py        # extraction du concept clinique via Qwen
├── recherche_snomed_B2.py                # recherche de candidats SNOMED via BioPortal
├── decision_match_C2.py                  # décision du match (Exact/Broader/Narrower/No Match)
├── generation_ttl_D2.py                  # génération du fragment RDF/Turtle
├── valider_ttl_D2.py                     # Validation syntaxique et sémantique du TTL
├── questionnaire_ontology.ttl             # Fragment RDF validé manuellement (référence)
├── questionnaire_ontology_AUTO_D2.ttl     # Fragment RDF généré automatiquement
└── .env                                  # Clé API BioPortal (non versionné)
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
  
