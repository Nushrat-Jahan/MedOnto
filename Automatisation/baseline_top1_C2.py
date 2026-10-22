
"""
Baseline : rang 1 de BioPortal


Ce rang 1 est comparé, item par item :
  - à la référence manuelle (identifiants SNOMED validés à la main, voir MANUAL),
  - au choix de C2 (resultats_C2_match.csv).

LECTURE SEULE : aucun appel réseau, aucun appel à Ollama, aucun fichier existant modifié.
Sortie : resultats_baseline_top1.csv. Le script refuse de tourner si ce fichier existe
déjà, pour ne jamais écraser un résultat.

"""

import csv
import json

INPUT_A2 = "resultats_A2.json"
INPUT_B2 = "resultats_B2_candidats.csv"
INPUT_C2 = "resultats_C2_match.csv"
OUTPUT_CSV = "resultats_baseline_top1.csv"

# Référence manuelle : (identifiant SNOMED validé à la main, label).
# None = aucun concept SNOMED retenu (Candidate Project Concept).
# À vérifier avec ton tableau de comparaison manuel avant de lancer.
MANUAL = {
    "phq2": ("366979004", "Depressed mood"),
    "phq4": ("84229001", "Fatigue"),
    "phq7": ("26329005", "Poor concentration"),
    "phq9": ("6471006", "Suicidal thoughts"),
    "gad1": ("48694002", "Anxiety"),
    "gad4": (None, "(no SNOMED CT concept - Candidate Project Concept)"),
    "gad6": ("55929007", "Feeling irritable"),
    "epw4": ("1367959007", "Excessive daytime sleepiness"),
    "epw8": ("1367959007", "Excessive daytime sleepiness"),
}

FIELDNAMES = [
    "Item ID", "Item text", "Concept (A2)",
    "Manual SNOMED ID", "Manual label",
    "Rank-1 SNOMED ID", "Rank-1 label", "Rank-1 correct",
    "Rank of manual concept among candidates",
    "C2 SNOMED ID", "C2 label", "C2 match", "C2 correct",
    "Comparison",
]


def read_csv(path: str) -> list:
    """Lit un CSV ; les en-têtes sont normalisés (minuscules, sans espaces autour)
    pour tolérer "SNOMED ID" / "Snomed ID" selon la version des fichiers."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [
            {(k or "").strip().lower(): (v or "") for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def compare(base_ok: bool, c2_ok: bool) -> str:
    if base_ok and c2_ok:
        return "Both correct"
    if base_ok:
        return "Rank 1 correct, C2 choice wrong"
    if c2_ok:
        return "C2 choice correct, rank 1 wrong"
    return "Both wrong"


def main() -> None:
    with open(INPUT_A2, encoding="utf-8") as f:
        items = json.load(f)

    candidates = {}
    for row in read_csv(INPUT_B2):
        candidates.setdefault(row["item id"].strip(), []).append(row)
    for rows in candidates.values():
        rows.sort(key=lambda r: int(r["rang"]))

    c2_by_item = {row["item id"].strip(): row for row in read_csv(INPUT_C2)}

    rows_out = []
    n_gold = base_gold_ok = c2_gold_ok = gold_in_list = 0
    n_none = base_none_ok = c2_none_ok = 0
    c2_better, c2_worse = [], []

    for item in items:
        iid = item["item_id"]
        if iid not in MANUAL:
            print(f"ATTENTION : {iid} absent de la référence manuelle (MANUAL), ignoré.")
            continue

        gold_id, gold_label = MANUAL[iid]
        gold_id = gold_id or ""

        cands = candidates.get(iid, [])
        top = cands[0] if cands else None
        top_id = top["snomed id"].strip() if top else ""
        top_label = top["preferred label"].strip() if top else ""

        c2 = c2_by_item.get(iid)
        c2_id = c2["snomed id"].strip() if c2 else ""
        c2_label = c2["preferred label"].strip() if c2 else ""
        c2_match = c2["match"].strip() if c2 else "(no C2 row)"

        base_ok = top_id == gold_id
        c2_ok = (c2 is not None) and c2_id == gold_id

        if gold_id:
            rank_of_gold = "absent"
            for r in cands:
                if r["snomed id"].strip() == gold_id:
                    rank_of_gold = r["rang"]
                    break
            n_gold += 1
            base_gold_ok += base_ok
            c2_gold_ok += c2_ok
            gold_in_list += rank_of_gold != "absent"
        else:
            rank_of_gold = "n/a"
            n_none += 1
            base_none_ok += base_ok
            c2_none_ok += c2_ok

        if c2_ok and not base_ok:
            c2_better.append(iid)
        if base_ok and not c2_ok:
            c2_worse.append(iid)

        rows_out.append({
            "Item ID": iid,
            "Item text": item.get("question", ""),
            "Concept (A2)": item.get("concept_proposed", ""),
            "Manual SNOMED ID": gold_id,
            "Manual label": gold_label,
            "Rank-1 SNOMED ID": top_id,
            "Rank-1 label": top_label,
            "Rank-1 correct": yes_no(base_ok),
            "Rank of manual concept among candidates": rank_of_gold,
            "C2 SNOMED ID": c2_id,
            "C2 label": c2_label,
            "C2 match": c2_match,
            "C2 correct": yes_no(c2_ok),
            "Comparison": compare(base_ok, c2_ok),
        })

    summary = [
        ("SUMMARY - items with a manual SNOMED concept", f"{n_gold}"),
        ("SUMMARY - rank 1 = manual concept", f"{base_gold_ok} / {n_gold}"),
        ("SUMMARY - C2 choice = manual concept", f"{c2_gold_ok} / {n_gold}"),
        ("SUMMARY - manual concept present among the candidates (ceiling for C2)",
         f"{gold_in_list} / {n_gold}"),
        ("SUMMARY - items without SNOMED concept: rank 1 returns nothing (correct abstention)",
         f"{base_none_ok} / {n_none}"),
        ("SUMMARY - items without SNOMED concept: C2 answers No Match (correct abstention)",
         f"{c2_none_ok} / {n_none}"),
        ("SUMMARY - items where C2 is better than rank 1", ", ".join(c2_better) or "none"),
        ("SUMMARY - items where C2 is worse than rank 1", ", ".join(c2_worse) or "none"),
    ]

    # Mode "x" : création exclusive, jamais d'écrasement d'un fichier existant
    try:
        out = open(OUTPUT_CSV, "x", newline="", encoding="utf-8-sig")
    except FileExistsError:
        raise SystemExit(
            f"{OUTPUT_CSV} existe déjà : le script ne l'écrase jamais. "
            "Renomme ou déplace l'ancien fichier, puis relance."
        )
    with out:
        writer = csv.DictWriter(out, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows_out)
        writer.writerow({})
        for label, value in summary:
            writer.writerow({"Item ID": label, "Comparison": value})

    print(f"Terminé. Résultats écrits dans {OUTPUT_CSV}\n")
    for label, value in summary:
        print(f"{label.replace('SUMMARY - ', '')} : {value}")


if __name__ == "__main__":
    main()