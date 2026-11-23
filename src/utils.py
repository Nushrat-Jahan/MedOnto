"""Fonctions utilitaires communes à toutes les parties : lecture et écriture de fichiers."""

from __future__ import annotations

import json
import re
import tomllib
from typing import Any, Protocol

from pypdf import PdfReader

# Mots vides anglais (the, of, and…), utilisés par le nettoyage ; liste vide si NLTK n'est pas installé
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except Exception as e:
    STOPWORDS = set()
    print(f"Warning: NLTK stopwords not loaded: {e}")


class SupportsToDict(Protocol):
    """Tout objet qui a une méthode to_dict(), comme les concepts du postprocess."""

    def to_dict(self) -> dict[str, Any]: ...


# ----------------------------------------------------------------------
# Chargement / écriture de fichiers
# ----------------------------------------------------------------------
def load_txt_file(text_path: str) -> str:
    """Renvoie le contenu d'un fichier texte."""
    with open(text_path, encoding="utf-8") as f:
        return f.read()


def load_pdf_file(text_path: str) -> str:
    """Renvoie le texte de toutes les pages d'un PDF."""
    reader = PdfReader(text_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_toml_file(toml_path: str) -> dict[str, Any]:
    """Renvoie le contenu d'un fichier de configuration TOML (réglages des modèles)."""
    with open(toml_path, "rb") as f:
        return tomllib.load(f)


def load_json_file(json_path: str):
    """Renvoie le contenu d'un fichier JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_file(json_path: str, data) -> None:
    """Écrit des données dans un fichier JSON lisible (indenté, accents conservés)."""
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ----------------------------------------------------------------------
# Parsing (ancien format : liste de concepts sous forme de chaînes)
# ----------------------------------------------------------------------
def parse_concept_list(raw: str) -> list[str] | None:
    """Lit une liste de concepts ["...", "..."] dans une réponse brute du LLM, sans doublons.

    Ancien format : plus utilisé par aucune partie. Renvoie None si la liste est illisible.
    """
    text = re.sub(r"```[a-zA-Z]*", "", raw.strip())
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    concepts: list[str] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        concepts.append(item)
    return concepts


def make_json_from_concepts(concepts: list[SupportsToDict]) -> list[dict[str, Any]]:
    """Transforme les concepts du postprocess en dictionnaires, prêts à écrire en JSON."""
    return [concept.to_dict() for concept in concepts]