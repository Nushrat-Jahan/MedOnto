"""
Étape D2 du pipeline automatique — Génération du fragment Turtle,
à but de COMPARAISON uniquement.

Génère questionnaire_ontology_AUTO_D2.ttl à partir de :
  - resultats_A2.json     (textes des items, concepts A2),
  - resultats_C2_match.csv (décisions de match ; IDs/labels issus de B2).

Structure identique à la référence manuelle fragment_ontologies_version2.ttl :
  Questionnaire --hasItem--> QuestionnaireItem --assesses--> ClinicalConcept
  QuestionnaireItem --hasResponseScale--> ResponseScale

Règle (identique à la référence manuelle) :
  - owl:sameAs UNIQUEMENT si le match C2 vaut "Exact" pour TOUTES les
    utilisations d'un même concept ;
  - Broader/Narrower : concept local + rdfs:comment (ID + label SNOMED),
    sans lien d'équivalence ;
  - No Match : concept local de classe CandidateProjectConcept, sans lien.

Contrôles :
  - aucun ID SNOMED non numérique n'est écrit (B2 ne produit que des IDs
    numériques issus des réponses BioPortal) ;
  - AUCUNE COLLISION D'IRI : si deux concepts distincts (labels différents)
    produisaient le même identifiant Turtle une fois nettoyé, ils seraient
    fusionnés en silence dans le graphe sans que rdflib ne le détecte à la
    validation. Le script échoue explicitement dans ce cas plutôt que de
    laisser une fusion silencieuse se produire.

Ce script n'écrit JAMAIS dans fragment_ontologies_version2.ttl ni dans
tout autre fichier existant.
"""

import csv
import json
import re

INPUT_A2 = "resultats_A2.json"
INPUT_C2 = "resultats_C2_match.csv"
OUTPUT_TTL = "questionnaire_ontology_AUTO_D2.ttl"
REF_MANUELLE = "fragment_ontologies_version2.ttl"

SNOMED_NS = "http://purl.bioontology.org/ontology/SNOMEDCT/"
Q_IRI = {"PHQ-9": "PHQ9", "GAD-7": "GAD7", "Epworth Sleepiness Scale": "ESS"}
SCALE_INFO = {
    "PHQ-9": ("PHQ9_ResponseScale", "PHQ-9 response scale (frequency over the past 2 weeks)"),
    "GAD-7": ("GAD7_ResponseScale", "GAD-7 response scale (frequency over the past 2 weeks)"),
    "Epworth Sleepiness Scale": ("ESS_ResponseScale", "ESS response scale (chance of dozing)"),
}


def turtle_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def local_iri(label: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9]", "", label.title())
    return name or "Concept"


def item_iri(item: dict) -> str:
    q = Q_IRI[item["questionnaire"]]
    num = re.search(r"(\d+)$", item["item_id"]).group(1)
    return f"{q}_Item{num}"


def main() -> None:
    with open(INPUT_A2, encoding="utf-8") as f:
        items = json.load(f)
    items_by_id = {it["item_id"]: it for it in items}

    decisions = {}
    with open(INPUT_C2, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            decisions[row["Item ID"].strip()] = row

    for iid in items_by_id:
        if iid not in decisions:
            raise SystemExit(f"C2 : pas de décision pour {iid} "
                             f"(resultats_C2_match.csv incomplet)")

    # --- Groupement des items par concept retenu ---
    # clé : ("snomed", id, label) si C2 a choisi un candidat,
    #      ("local", concept_a2_minuscule, concept_a2) si No Match
    def key_for(iid: str):
        d = decisions[iid]
        if d["SNOMED ID"].strip():
            return ("snomed", d["SNOMED ID"].strip(), d["Preferred label"].strip())
        concept_a2 = items_by_id[iid].get("concept_proposed", "").strip()
        return ("local", concept_a2.lower(), concept_a2)

    groups = {}
    for iid in items_by_id:
        groups.setdefault(key_for(iid), []).append(
            (iid, decisions[iid]["Match"].strip()))

    concept_iri = {}
    for key in groups:
        kind, ident, label = key
        if kind == "local" and not label:
            raise SystemExit(f"D2 : concept A2 vide pour un No Match (clé {key!r})")
        if kind == "local":
            concept_iri[key] = "Candidate_" + local_iri(label)
        else:
            concept_iri[key] = local_iri(label)

    # Garde-fou : deux concepts distincts ne doivent jamais produire le même IRI.
    # Sans ce contrôle, une fusion silencieuse serait possible (deux labels
    # différents nettoyés vers le même identifiant Turtle), invisible à la
    # validation rdflib puisque le fichier resterait syntaxiquement correct.
    iri_to_keys = {}
    for key, iri in concept_iri.items():
        iri_to_keys.setdefault(iri, []).append(key)
    collisions = {iri: keys for iri, keys in iri_to_keys.items() if len(keys) > 1}
    if collisions:
        raise SystemExit(f"D2 : collision d'IRI entre concepts distincts -> {collisions}")

    out = []
    out.append("# Généré automatiquement par le pipeline A2 (Qwen) -> B2 (BioPortal) -> C2 (Qwen) -> D2.")
    out.append("# NON validé manuellement. À comparer avec " + REF_MANUELLE + " (référence manuelle).")
    out.append('# Règle : owl:sameAs uniquement si le match C2 vaut "Exact" pour toutes les utilisations du concept.')
    out.append("")
    out.append("@prefix :     <http://example.org/questionnaire-ontology#> .")
    out.append("@prefix owl:  <http://www.w3.org/2002/07/owl#> .")
    out.append("@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .")
    out.append("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .")
    out.append("@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .")
    out.append("")
    out.append("<http://example.org/questionnaire-ontology> a owl:Ontology ;")
    out.append('    rdfs:label "Questionnaire ontology fragment AUTO (PHQ-9, GAD-7, ESS) - pipeline A2D2"@en .')
    out.append("")
    out.append("########################################################################")
    out.append("# Classes")
    out.append("########################################################################")
    out.append(':Questionnaire     a owl:Class ; rdfs:label "Questionnaire"@en .')
    out.append(':QuestionnaireItem a owl:Class ; rdfs:label "Questionnaire item"@en .')
    out.append(':ClinicalConcept   a owl:Class ; rdfs:label "Clinical concept"@en .')
    out.append(':ResponseScale     a owl:Class ; rdfs:label "Response scale"@en .')
    out.append(':CandidateProjectConcept a owl:Class ;')
    out.append('    rdfs:subClassOf :ClinicalConcept ;')
    out.append('    rdfs:label "Candidate project concept"@en ;')
    out.append('    rdfs:comment "Clinical concept with no suitable SNOMED CT match found."@en .')
    out.append("")
    out.append("########################################################################")
    out.append("# Propriétés")
    out.append("########################################################################")
    out.append(':hasItem a owl:ObjectProperty ;')
    out.append('    rdfs:label "has item"@en ;')
    out.append('    rdfs:domain :Questionnaire ;')
    out.append('    rdfs:range  :QuestionnaireItem .')
    out.append("")
    out.append(':assesses a owl:ObjectProperty ;')
    out.append('    rdfs:label "assesses"@en ;')
    out.append('    rdfs:domain :QuestionnaireItem ;')
    out.append('    rdfs:range  :ClinicalConcept .')
    out.append("")
    out.append(':hasResponseScale a owl:ObjectProperty ;')
    out.append('    rdfs:label "has response scale"@en ;')
    out.append('    rdfs:domain :QuestionnaireItem ;')
    out.append('    rdfs:range  :ResponseScale .')
    out.append("")
    out.append(':itemId a owl:DatatypeProperty ;')
    out.append('    rdfs:domain :QuestionnaireItem ;')
    out.append('    rdfs:range  xsd:string .')
    out.append("")
    out.append(':minScore a owl:DatatypeProperty ;')
    out.append('    rdfs:domain :ResponseScale ;')
    out.append('    rdfs:range  xsd:integer .')
    out.append("")
    out.append(':maxScore a owl:DatatypeProperty ;')
    out.append('    rdfs:domain :ResponseScale ;')
    out.append('    rdfs:range  xsd:integer .')
    out.append("")
    out.append(':responseLabels a owl:DatatypeProperty ;')
    out.append('    rdfs:domain :ResponseScale ;')
    out.append('    rdfs:range  xsd:string .')
    out.append("")
    out.append("########################################################################")
    out.append("# Questionnaires")
    out.append("########################################################################")
    for qname, qiri in Q_IRI.items():
        members = [it for it in items if it["questionnaire"] == qname]
        has_items = ", ".join(f":{item_iri(it)}" for it in members)
        out.append(f":{qiri} a :Questionnaire ;")
        out.append(f'    rdfs:label "{qname}"@en ;')
        out.append(f"    :hasItem {has_items} .")
        out.append("")
    out.append("########################################################################")
    out.append("# Échelles de réponse")
    out.append("########################################################################")
    for qname, (siri, slabel) in SCALE_INFO.items():
        member = next(it for it in items if it["questionnaire"] == qname)
        out.append(f":{siri} a :ResponseScale ;")
        out.append(f'    rdfs:label "{turtle_escape(slabel)}"@en ;')
        out.append("    :minScore 0 ; :maxScore 3 ;")
        out.append(f'    :responseLabels "{turtle_escape(member["scale"])}" .')
        out.append("")
    out.append("########################################################################")
    out.append("# Items")
    out.append("########################################################################")
    for it in items:
        iid = it["item_id"]
        key = key_for(iid)
        siri = SCALE_INFO[it["questionnaire"]][0]
        out.append(f":{item_iri(it)} a :QuestionnaireItem ;")
        out.append(f'    :itemId "{iid}" ;')
        out.append(f'    rdfs:label "{turtle_escape(it["question"])}"@en ;')
        out.append(f"    :assesses :{concept_iri[key]} ;")
        out.append(f"    :hasResponseScale :{siri} .")
        out.append("")
    out.append("########################################################################")
    out.append("# Concepts cliniques (IDs issus de B2/BioPortal ; choix issus de C2)")
    out.append("# owl:sameAs uniquement pour les concepts dont TOUS les items sont Exact.")
    out.append("########################################################################")
    for key, usages in groups.items():
        kind, ident, label = key
        iri = concept_iri[key]
        matches = [m for _, m in usages]
        iids = [iid for iid, _ in usages]
        out.append("")
        if kind == "local":
            out.append(f":{iri} a :ClinicalConcept, :CandidateProjectConcept ;")
            out.append(f'    rdfs:label "{turtle_escape(label)}"@en ;')
            out.append(f'    rdfs:comment "No SNOMED CT match found for the concept proposed on item(s) '
                       f'{turtle_escape(", ".join(iids))} (pipeline automatique, non validé manuellement)."@en .')
            continue
        if not ident.isdigit():
            raise SystemExit(f"D2 : ID SNOMED non numérique {ident!r} refusé (doit venir de B2/BioPortal)")
        if all(m == "Exact" for m in matches):
            out.append(f":{iri} a :ClinicalConcept ;")
            out.append(f'    rdfs:label "{turtle_escape(label)}"@en ;')
            out.append(f"    owl:sameAs <{SNOMED_NS}{ident}> .")
        else:
            out.append(f":{iri} a :ClinicalConcept ;")
            out.append(f'    rdfs:label "{turtle_escape(label)}"@en ;')
            out.append(f'    rdfs:comment "Match(s) {turtle_escape(", ".join(sorted(set(matches))))} avec SNOMED CT '
                       f'{ident} ({turtle_escape(label)}) pour {turtle_escape(", ".join(iids))}; '
                       'pas d\'équivalence (pipeline automatique, non validé manuellement)."@en .')

    with open(OUTPUT_TTL, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

    n_sameas = sum(1 for key, usages in groups.items()
                   if key[0] == "snomed" and all(m == "Exact" for _, m in usages))
    print(f"Écrit : {OUTPUT_TTL} ({n_sameas} owl:sameAs)")
    print(f"À comparer avec : {REF_MANUELLE} (référence manuelle, non modifiée)")


if __name__ == "__main__":
    main()