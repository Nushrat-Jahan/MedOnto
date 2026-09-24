import json
import re
import sys
from pathlib import Path

OUT_DIR = Path("outputs")
PROJECT_PREFIX = "http://example.org/ontology#"

HEADER = f"""@prefix proj: <{PROJECT_PREFIX}> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
"""

# match: Exact/Broader/Narrower -> relation SKOS correspondante
MATCH_PREDICATE = {
    "Exact": "skos:exactMatch",
    "Broader": "skos:broadMatch",
    "Narrower": "skos:narrowMatch",
}


def slug(text: str) -> str:
    """Transforme un texte libre en identifiant Turtle valide (proj:CeciEstValide)."""
    return re.sub(r"[^A-Za-z0-9]", "", text)


def esc(text: str) -> str:
    """Échappe les guillemets/backslash/retours à la ligne pour un littéral Turtle."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def build_turtle(results: list[dict]) -> tuple[str, int]:
    """Construit le texte Turtle pour toutes les lignes validées 'Correct'. Renvoie (texte, nb_triplets)."""
    blocks = []
    triple_count = 0

    for row in results:
        if row.get("validation_status") != "Correct":
            continue

        concept = slug(row["normalized_concept"])
        category = slug(row["semantic_category"])

        lines = [
            f"proj:{concept} rdfs:subClassOf proj:{category} ;",
            f'    rdfs:label "{esc(row["normalized_concept"])}" ;',
        ]
        triple_count += 2

        if row.get("relation"):
            variable = slug(row["variable"])
            relation = slug(row["relation"])
            lines.append(f'    rdfs:comment "{esc(row.get("evidence", ""))}" ;')
            lines.append(f"    proj:{relation} proj:{variable} .")
            triple_count += 2
        else:
            # dernière ligne du bloc : remplacer le ";" final par un "."
            lines[-1] = lines[-1][:-2] + "."
            triple_count += 1

        blocks.append("\n".join(lines))

        match = row.get("match")
        predicate = MATCH_PREDICATE.get(match)
        if predicate and row.get("identifier_uri"):
            uri = row["identifier_uri"]
            blocks.append(f'proj:{concept} {predicate} <{uri}> .')
            if row.get("preferred_label"):
                blocks.append(f'<{uri}> rdfs:label "{esc(row["preferred_label"])}" .')
                triple_count += 1
            triple_count += 1

    return "\n\n".join(blocks), triple_count


def main(results_path: Path) -> None:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    rows = data["results"]

    counts = {"Correct": 0, "Incorrect": 0, "Uncertain": 0}
    for row in rows:
        status = row.get("validation_status", "Uncertain")
        counts[status] = counts.get(status, 0) + 1
    print(f"Statuts de validation: {counts}")

    body, triple_count = build_turtle(rows)
    if triple_count == 0:
        print("Aucune ligne 'Correct' trouvée -> pas de fragment généré. "
              "Édite 'validation_status' dans le fichier de résultats d'abord.")
        return

    out_path = OUT_DIR / f"ontologie_fragment_{results_path.stem}.ttl"
    out_path.write_text(HEADER + "\n" + body + "\n", encoding="utf-8")
    print(f"{triple_count} triplets -> '{out_path}'")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/build_ontology.py outputs/qwen_resultats_<date>_<id>.json")
        sys.exit(1)
    main(Path(sys.argv[1]))
