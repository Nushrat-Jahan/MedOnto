import json
import re
import tomllib
from typing import Any

def load_txt_file(text_path:str) -> str: 
    with open(text_path, encoding="utf-8") as f:
        text:str = f.read()

    return text 



def load_toml_file(toml_path:str) -> dict[str,any]: 
    with open(toml_path,"rb") as f:
        toml:dict[str,any] = tomllib.load(f)

    return toml


def parse_concept_list(raw: str) -> list[str] | None:
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


_TRIPLE_OBJECT_RE = re.compile(
    r'\{\s*"subject"\s*:\s*"(?:[^"\\]|\\.)*"\s*,\s*'
    r'"relation"\s*:\s*"(?:[^"\\]|\\.)*"\s*,\s*'
    r'"object"\s*:\s*"(?:[^"\\]|\\.)*"\s*\}'
)


def _triplet_from_item(item: Any) -> tuple[str, str, str] | None:
    if isinstance(item, dict):
        subject = str(item.get("subject", "")).strip()
        relation = str(item.get("relation", "")).strip()
        obj = str(item.get("object", "")).strip()
    elif isinstance(item, list) and len(item) == 3:
        subject, relation, obj = (str(part).strip() for part in item)
    else:
        return None
    if not subject or not relation or not obj:
        return None
    return (subject, relation, obj)


def parse_triplet_list(raw: str) -> list[tuple[str, str, str]] | None:
    text = re.sub(r"```[a-zA-Z]*", "", raw.strip())
    candidates: list[Any] | None = None
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            candidates = data
    if candidates is None:
        candidates = []
        for match in _TRIPLE_OBJECT_RE.finditer(text):
            try:
                candidates.append(json.loads(match.group(0)))
            except json.JSONDecodeError:
                continue
    triplets: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in candidates:
        triplet = _triplet_from_item(item)
        if triplet is None or triplet in seen:
            continue
        seen.add(triplet)
        triplets.append(triplet)
    return triplets