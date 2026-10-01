"""
Étape D2 (validation) de questionnaire_ontology_AUTO_D2.ttl.
Copie adaptée de validate_ttl.py, qui reste non modifié.

Contrôles :
  1. Syntaxe : le fichier parse-t-il avec rdflib ?
  2. Structure : chaque item a exactement un questionnaire (hasItem), un
     concept (assesses), une échelle (hasResponseScale) ; aucun item n'est
     aussi typé ClinicalConcept.
  3. owl:sameAs conforme à la RÈGLE PAR GROUPE appliquée par D2 : un ID
     SNOMED n'a droit à owl:sameAs QUE SI toutes les lignes de
     resultats_C2_match.csv qui l'utilisent ont Match == "Exact" (et non
     une vérification ligne par ligne, qui donnerait de faux positifs dès
     qu'un même concept est partagé par plusieurs items à matchs différents,
     comme epw4/epw8 dans le fragment manuel).
  4. Aucun ID SNOMED écrit dans le graphe n'est absent de
     resultats_C2_match.csv (donc jamais un ID que C2 n'a pas retenu).
"""

import csv
from rdflib import Graph, Namespace, RDF, OWL

TTL_FILE = "questionnaire_ontology_AUTO_D2.ttl"
MATCH_CSV = "resultats_C2_match.csv"
EX = Namespace("http://example.org/questionnaire-ontology#")

# ---------- 1. Charger la décision de C2 (référence dynamique) ----------
matches_by_id = {}
with open(MATCH_CSV, encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        sid = row["SNOMED ID"].strip()
        if sid:
            matches_by_id.setdefault(sid, []).append(row["Match"].strip())

# Un ID a droit à owl:sameAs seulement si TOUTES ses utilisations sont Exact
# (règle exacte appliquée par generation_ttl_D2.py)
ids_expecting_sameas = {sid for sid, ms in matches_by_id.items() if all(m == "Exact" for m in ms)}

# ---------- 2. Syntaxe ----------
g = Graph()
g.parse(TTL_FILE, format="turtle")
print("RDF/Turtle parsed successfully.")
print("Number of triples:", len(g))

problems = []

# ---------- 3. Structure des items ----------
items = set(g.subjects(RDF.type, EX.QuestionnaireItem))
print("Items:", len(items))
for item in sorted(items):
    name = item.split("#")[-1]
    if len(list(g.subjects(EX.hasItem, item))) != 1:
        problems.append(f"{name}: nombre de hasItem incorrect")
    if len(list(g.objects(item, EX.assesses))) != 1:
        problems.append(f"{name}: nombre de assesses incorrect")
    if len(list(g.objects(item, EX.hasResponseScale))) != 1:
        problems.append(f"{name}: nombre de hasResponseScale incorrect")
    if (item, RDF.type, EX.ClinicalConcept) in g:
        problems.append(f"{name}: item aussi typé ClinicalConcept")

# ---------- 4. owl:sameAs conforme à la règle PAR GROUPE de D2 ----------
sameas = list(g.subject_objects(OWL.sameAs))
print("owl:sameAs triples:", len(sameas))
sameas_ids = set()
for s, o in sameas:
    sid = str(o).rsplit("/", 1)[-1]
    sameas_ids.add(sid)
    print(f"  {s.split('#')[-1]}  sameAs  {o}")
    if sid not in matches_by_id:
        problems.append(f"{s.split('#')[-1]}: ID {sid} absent de resultats_C2_match.csv "
                         f"(URI non issue du pipeline)")
    elif sid not in ids_expecting_sameas:
        problems.append(f"{s.split('#')[-1]}: owl:sameAs vers {sid}, mais toutes les "
                         f"utilisations C2 ne sont pas Exact ({matches_by_id[sid]})")

# Un concept entièrement Exact dans C2 devrait avoir un owl:sameAs dans le TTL
missing = ids_expecting_sameas - sameas_ids
if missing:
    problems.append(f"Concepts entièrement Exact dans C2 sans owl:sameAs correspondant "
                     f"dans le TTL : {missing}")

print("\nContrôles :", "OK" if not problems else "PROBLÈMES")
for p in problems:
    print(" -", p)