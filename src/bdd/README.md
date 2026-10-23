# Partie données de cohorte structurées (`src/bdd`)

Cette partie transforme **n'importe quelle base de données structurée** (un tableau de variables,
au format CSV ou Excel) en contenu d'ontologie aligné sur **SNOMED CT**, avec l'aide de Qwen2.5.
Rien n'est codé en dur : les noms de colonnes, le séparateur, l'encodage et le format sont détectés
à partir du fichier lui-même. Le même pipeline fonctionne donc sur la cohorte PR/BRI et sur les
modèles de résultats biologiques BIO.

## L'avancement en un coup d'œil

| Étape | Ce qu'elle fait | Fichier | État |
|---|---|---|---|
| 1. Variables | Lit le tableau et décrit chaque colonne (libellé, unité, type, valeurs, intervalle) | `extract_variables.py` | ✅ fait |
| 2. Concepts | Qwen propose un concept proche de SNOMED CT pour chaque variable | `extract_concepts.py` | ✅ fait |
| 3. Recherche SNOMED CT | Cherche chaque concept proposé dans SNOMED CT (API BioPortal) | `search_snomed.py` | ✅ fait |
| 4. Décision du match | Choisit le bon candidat : Exact / Broader / Narrower / No Match | — | ⬜ prochaine étape |
| 5. Fragment d'ontologie | Écrit le fragment Turtle (RDF/OWL) | — | ⬜ à faire |
| 6. Évaluation | Mesure la qualité du résultat | — | ⬜ à faire |

Les étapes 1 à 3 s'enchaînent avec une seule commande (voir ci-dessous). Les étapes 3 à 5
reprennent le travail de la partie questionnaires (`src/questions/`), adapté aux variables d'un
tableau.

## Lancement

**Prérequis :**

- Python 3.11 (version testée), avec les bibliothèques du pipeline :

  ```bash
  python -m pip install pandas openpyxl ollama python-dotenv
  ```

- [Ollama](https://ollama.com) installé en local, avec le modèle `qwen2.5:7b`
  (`ollama pull qwen2.5:7b`). Qwen demande environ 5 Go de mémoire libre ;
- une clé de l'API BioPortal dans un fichier `.env` à la racine du projet
  (`BIOPORTAL_API_KEY=...`, jamais versionné).

**Lancement**, depuis la racine du projet :

```bash
python main.py "data/<ma_base>.csv"
python main.py "data/<ma_base>.xlsx" --label-row    # si la ligne 2 contient les libellés des variables
```

Sous Windows, quand plusieurs versions de Python sont installées, `python` peut lancer une version
qui n'a pas ces bibliothèques (erreur `ModuleNotFoundError: No module named 'pandas'`). Il suffit
alors de choisir la version avec le lanceur `py` :

```bash
py -3.11 main.py "data/<ma_base>.xlsx" --label-row
```

Les bases restent dans `data/`, qui n'est pas versionné. Sur un ordinateur sans carte graphique,
compter environ 3 min 30 pour une base de 14 variables.

Tous les résultats d'une base sont écrits dans son propre dossier, `outputs/<nom_base>_<id>/`, où
`<id>` est calculé à partir des noms des colonnes :

| Fichier | Étape | Contenu |
|---|---|---|
| `variables.json` | 1 | une description par colonne, et la documentation de la base (feuille README) |
| `concepts_<id du prompt>.json` | 2 | pour chaque variable : le prompt envoyé, la réponse brute de Qwen et le concept lu |
| `candidates_<id du prompt>.json` | 3 | pour chaque variable : le terme cherché et les candidats SNOMED CT (ID, libellé, hiérarchie) |

`<id du prompt>` est une empreinte du prompt système : modifier le prompt crée de nouveaux fichiers
au lieu d'écraser les résultats précédents. On peut ainsi comparer les versions du prompt.

## Les étapes en détail

**1. Variables (`extract_variables.py`).** Lit un CSV (encodage et séparateur détectés) ou un
Excel (première feuille). Avec `--label-row`, la ligne 2 donne les libellés et les unités
(`Bio. : CRP (mg/L)` → unité `mg/L`). Le type de chaque colonne est déduit **des données**
(identifiant, binaire, entier, décimal, catégoriel…) ou, pour un modèle vide, **de son libellé**
(`(ON)` → Oui/Non, `date` → date, une unité → numérique). La feuille README d'un fichier Excel est
recopiée pour servir de contexte à Qwen. Les descriptions sont rédigées par le script, et non par
le LLM, pour ne pas introduire d'inventions avant même l'extraction.

**2. Concepts (`extract_concepts.py`).** Un appel à Qwen par variable (température 0, réponse en
JSON imposée par Ollama). La sortie suit **le même format que la partie questionnaires** :
`item_id`, `concept_proposed`, `category`, `relation`, `evidence`, `status`. Les identifiants
(ex. numéro de patient) ne sont pas envoyés. Le script ne corrige rien : les réponses hors format
sont seulement signalées.

**3. Recherche SNOMED CT (`search_snomed.py`).** Adapté de `src/questions/recherche_snomed_B2.py`.
Pour chaque concept : recherche exacte d'abord, puis recherche large (10 candidats) si rien n'est
trouvé. Seuls les vrais identifiants SNOMED CT renvoyés par l'API sont gardés : aucun LLM ne peut
inventer d'identifiant. Chaque candidat garde sa hiérarchie SNOMED CT (`procedure`,
`observable entity`…) pour l'étape 4.

## Premiers résultats : modèle BIO_EMPILEE (14 variables, données synthétiques et vides)

| Étape | Résultat |
|---|---|
| 1 | 13 variables sur 14 typées à partir de leur libellé (seule `VISIT` reste sans type) |
| 2 | 13 réponses sur 13 en JSON valide et au format attendu ; la plupart des termes sont justes et en anglais, même à partir de libellés en français |
| 3 | **6 variables sur 13** obtiennent le bon concept SNOMED CT en premier (CRP `55235003`, créatinine `70901006`, neutrophiles, lymphocytes, plaquettes, GGT) ; 1 de plus l'a dans sa liste (ALAT) |

**Temps d'exécution :** 3,4 min pour les étapes 1 à 3, sur un ordinateur portable sans carte
graphique (16 Go de mémoire). Presque tout ce temps vient de Qwen (environ 14 s par variable,
surtout pour écrire sa réponse) ; les 13 recherches dans l'API prennent 13 s.

**Ce que montrent ces résultats :**

- quand Qwen propose le terme exact de SNOMED CT, la recherche trouve le concept du premier coup ;
- tous les échecs de la recherche viennent **du terme proposé à l'étape 2** : *leukocyte count* au
  lieu de *white blood cell count*, ASAT confondue avec ALAT (aspartate et alanine
  aminotransférase), *CRP elevation* au lieu de *C-reactive protein abnormal*. Avec les bons
  termes, la recherche exacte trouve les trois ;
- les catégories de Qwen ne sont pas fiables (examens de laboratoire classés `observable_entity`
  au lieu de `procedure`), et son statut vaut toujours `existing_concept` : c'est l'API, et non
  Qwen, qui doit trancher.

## Limites connues

- **Le format large n'est pas encore regroupé :** dans `BIO_NON_EMPILEE`, `V0_BIO_CRP` …
  `V4_BIO_CRP` sont un même concept mesuré à 5 visites ; sans regroupement, Qwen reçoit 60 appels
  au lieu de 12.
- **Les libellés sont en français** et le prompt en anglais : Qwen traduit, avec parfois des
  erreurs (ASAT/ALAT).
- **Les relations valent le plus souvent `null`**, volontairement : en semaine 1, imposer une
  relation poussait Qwen à en inventer des génériques. Les relations de structure
  (patient → visite → mesure → unité) seront produites par le script à l'étape 5. Les relations
  entre variables (ex. *CRP augmentée* est une interprétation de *CRP*) demanderont une étape à
  part, car Qwen ne voit qu'une variable à la fois.
- La cohorte PR/BRI passe par l'étape 1 ; les étapes 2 et 3 n'ont pour l'instant été lancées que
  sur le modèle BIO.

## Prochaines étapes

1. **Étape 4, décision du match**, adaptée de `src/questions/decision_match_C2.py`. Qwen n'est pas
   appelé quand la recherche exacte ne renvoie qu'un seul candidat.
2. **Étape 5, fragment Turtle**, avec les alignements SNOMED CT et les relations de structure.
3. **Étape 6, évaluation** par rapport à la référence manuelle de la semaine 1 (10 variables
   PR/BRI alignées et validées à la main).
4. **Améliorer la recherche** : demander à Qwen 2 ou 3 termes possibles par variable, puis filtrer
   la recherche large par hiérarchie SNOMED CT quand la catégorie est fiable ; regrouper les visites
   du format large.

## Points à décider avec les autres parties

- **Le même modèle pour tous :** `qwen2.5:7b` (Ollama) ici et dans la partie questionnaires,
  `Qwen2.5-1.5B-Instruct` dans la partie texte ; sinon, les résultats ne sont pas comparables.
- **Une liste commune de catégories :** la liste actuelle est pensée pour les questionnaires et
  n'a pas `procedure`, la hiérarchie SNOMED CT des examens de laboratoire.
- **Une recherche SNOMED CT partagée** dans `src/verification.py`, et un `requirements.txt` commun.
