import re 
import numpy as np 
from src.spanextraction import Triplet
import json 
from typing import Set


try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except Exception as e:
    raise ImportError(
        "NLTK required. Install: pip install nltk\n"
        "Then: python -c \"import nltk; nltk.download('stopwords')\""
    ) from e


_BLACKLIST = None


def load_blacklist(json_path: str) -> Set[str]:
    with open(json_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        return set(term.lower().strip() for term in data)
    elif isinstance(data, dict) and 'blacklist' in data:
        return set(term.lower().strip() for term in data['blacklist'])
    else:
        raise ValueError("Invalid blacklist format")


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def is_punct_or_space(s: str) -> bool:
    s = (s or "").strip()
    return not s or re.fullmatch(r"[\W_]+", s) is not None


def is_anon(s: str) -> bool:
    s = (s or "").strip()
    return "___" in s or re.fullmatch(r"_+", s) is not None


def is_stopword(s: str) -> bool:
    t = (s or "").strip().lower()
    if not t or re.fullmatch(r"[a-z]+", t) is None:
        return False
    return t in STOPWORDS


def is_digits_only(s: str) -> bool:
    return re.fullmatch(r"\d+", (s or "").strip()) is not None


def is_blacklisted(s: str) -> bool:
    if _BLACKLIST is None:
        return False
    return (s or "").strip().lower() in _BLACKLIST


def is_lab_value_pattern(s: str) -> bool:
    t = (s or "").strip().lower()
    if re.fullmatch(r"[a-z]{2,10}-\d+\.?\d*", t):
        return True

    if re.fullmatch(r"[a-z]{1,6}-[a-z]{1,6}-\d+\.?\d*", t):
        return True
    return False


def should_skip(s: str) -> bool:
    return (
        is_punct_or_space(s)
        or is_anon(s)
        or is_stopword(s)
        or is_digits_only(s)
        or is_blacklisted(s)
        or is_lab_value_pattern(s)
    )




def clean_outputs(outputs: list[Triplet]) -> list[Triplet]:
    cleaned:list[Triplet] = []
    for output in outputs:
        if output is None:
            continue
        relation = normalize(output.relation)
        subject = normalize(output.subject)
        object = normalize(output.object)

        if should_skip(relation) or should_skip(subject) or should_skip(object):
            continue

        cleaned.append(Triplet(subject,relation,object))

    return cleaned

