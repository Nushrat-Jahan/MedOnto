import re
import json
from typing import NamedTuple, Optional, Set, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from functools import lru_cache
from transformers import AutoTokenizer, AutoModel

from src.utils import STOPWORDS

_BLACKLIST = None

CATEGORIES = {"finding", "symptom", "disorder", "behavior",
              "observable_entity", "assessment_scale"}
PREDICATES = {"is_a", "has_associated_finding", "has_interpretation",
              "measures", "is_manifestation_of"}
STATUSES = {"existing_concept", "candidate_project_concept"}
MAX_EVIDENCE_WORDS = 25


class Concept(NamedTuple):
    item_id: str
    concept_proposed: str
    category: str
    predicate: Optional[str]
    target_concept: Optional[str]
    evidence: str
    status: str

    @property
    def subject(self) -> str:
        return self.concept_proposed

    @property
    def relation(self) -> str:
        return self.predicate or ""

    @property
    def object(self) -> str:
        return self.target_concept or ""

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "concept_proposed": self.concept_proposed,
            "category": self.category,
            "relation": None if self.predicate is None else {
                "predicate": self.predicate,
                "target_concept": self.target_concept,
            },
            "evidence": self.evidence,
            "status": self.status,
        }


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


def _parse_output(raw: Union[str, dict, None]) -> Optional[dict]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def _truncate_words(text: str, n: int = MAX_EVIDENCE_WORDS) -> str:
    words = (text or "").split()
    if len(words) <= n:
        return " ".join(words)
    return " ".join(words[:n]).rstrip(",;:") + "."


def _to_concepts(raw: Union[str, dict, None]) -> list[Concept]:
    data = _parse_output(raw)
    if data is None:
        return []

    item_id = str(data.get("item_id") or "").strip()
    category = normalize(data.get("category"))
    status = normalize(data.get("status"))
    evidence = _truncate_words((data.get("evidence") or "").strip())

    if not item_id or category not in CATEGORIES or status not in STATUSES or not evidence:
        return []

    predicate, target = None, None
    rel = data.get("relation")
    if isinstance(rel, dict):
        p = normalize(rel.get("predicate"))
        t = normalize(str(rel.get("target_concept") or "").replace("_", " "))
        if p in PREDICATES and not should_skip(t):
            predicate, target = p, t

    raw_concepts_str = str(data.get("concept_proposed") or "")
    split_concepts = [c.strip() for c in raw_concepts_str.split(",") if c.strip()]

    extracted_concepts: list[Concept] = []
    for raw_c in split_concepts:
        concept = normalize(raw_c.replace("_", " "))
        
        if should_skip(concept):
            continue

        rel_p = predicate if target != concept else None
        rel_t = target if target != concept else None

        extracted_concepts.append(
            Concept(item_id, concept, category, rel_p, rel_t, evidence, status)
        )

    return extracted_concepts


def _clean_list(outputs: list) -> list[Concept]:
    cleaned: list[Concept] = []
    for raw in outputs:
        concepts = _to_concepts(raw)
        cleaned.extend(concepts)
    return cleaned


@lru_cache(maxsize=1)
def _get_model():
    model_id = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoder = AutoModel.from_pretrained(model_id)
    encoder.eval()
    return tokenizer, encoder


def _compute_sentence_embeddings(concepts: list[Concept]) -> Tensor:
    tokenizer, encoder = _get_model()

    texts = [c.concept_proposed for c in concepts]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = encoder(**inputs)

    token_embeddings = outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1)
    sentence_embedding = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    return F.normalize(sentence_embedding, p=2, dim=1)


def create_mask(sentence_embeddings: Tensor, threshold: float = 0.85) -> Tensor:
    similarities = sentence_embeddings @ sentence_embeddings.T
    n = similarities.shape[0]
    mask = torch.ones(n, dtype=torch.bool, device=sentence_embeddings.device)

    for i in tqdm(range(n), desc="Cleaning synonyms"):
        if not mask[i]:
            continue
        duplicates = similarities[i, i + 1:] > threshold
        mask[i + 1:][duplicates] = False

    return mask


def _clean(outputs: list, threshold: float = 0.85) -> list[Concept]:
    cleaned = _clean_list(outputs)
    if not cleaned:
        return []

    cleaned.sort(key=lambda c: c.status != "existing_concept")

    seen, unique = set(), []
    for c in cleaned:
        key = (c.concept_proposed, c.category)
        if key not in seen:
            seen.add(key)
            unique.append(c)

    embeddings = _compute_sentence_embeddings(unique)
    mask = create_mask(embeddings, threshold)

    return [c for c, keep in zip(unique, mask.tolist()) if keep]


def postprocess(lists_outputs: list[list], threshold: float = 0.85) -> list[list[Concept]]:
    return [_clean(group, threshold=threshold) for group in lists_outputs]