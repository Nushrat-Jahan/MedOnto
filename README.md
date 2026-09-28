# MedOnto
LLM-assisted ontology construction from heterogeneous health data.

## Données structurées (bases tabulaires)

Pipeline automatisé qui transforme les variables d'une base de données structurée (CSV…) en
concepts normalisés, catégories sémantiques et relations, puis en fragment d'ontologie aligné
sur **SNOMED CT**. Il doit fonctionner avec **n'importe quelle base fournie** : aucun nom de
colonne, séparateur ou format n'est codé en dur.

### Organisation prévue

```text
main.py            ← point d'entrée unique : exécute les étapes du pipeline dans l'ordre
src/               ← une étape = un module de fonctions, appelées par main.py
data/              ← bases de données fournies (non versionnées)
outputs/
  <nom_base>_<id>/ ← un dossier de résultats par base
```

- `main.py` **importe** les fonctions des modules de `src/` et les appelle dans l'ordre ;
  chaque fonction reste utilisable et testable séparément.
- La base est passée **en argument** : le pipeline peut ainsi être lancé sans intervention,
  y compris sur plusieurs bases à la suite.

  ```bash
  python main.py data/ma_base.csv
  ```

- Chaque base reçoit un **identifiant** (empreinte de son contenu) : ses résultats sont rangés
  dans `outputs/<nom_base>_<id>/`, ce qui permet de tester plusieurs bases sans les mélanger.
- Chaque étape écrit sa sortie en **JSON**, lue par l'étape suivante : on peut relancer le
  pipeline à partir d'une étape sans refaire les précédentes (notamment les appels au LLM).

### Étapes du pipeline

1. **Extraction des variables** : lecture de la base et description de chaque variable
   (nom, type, valeurs, exemple…).
2. **Description des variables** : rédaction automatique de la description de chaque variable
   par le script, à partir des données elles-mêmes (sans passer par le LLM).
3. **Construction du prompt** à partir des variables extraites.
4. **Exécution du prompt** avec Qwen 2.5 7B en local via Ollama et récupération de la sortie JSON.
5. **Validation des concepts** via une API SNOMED CT.
6. **Construction du fragment d'ontologie** (Turtle) à partir des concepts validés.
7. **Évaluation** de la sortie à l'aide de métriques.

Le pipeline est développé étape par étape ; `main.py` est complété au fur et à mesure que les
étapes sont disponibles.

### Lancement

Les scripts se lancent depuis la racine du projet. Prérequis : Python 3, `pandas`, `ollama`, et
le modèle `qwen2.5:7b` installé dans Ollama.
