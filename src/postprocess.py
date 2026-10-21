import re 
import numpy as np 
import json 
import torch
from typing import NamedTuple, Set
from src.utils import STOPWORDS
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from functools import lru_cache

_BLACKLIST = None


class Triplet(NamedTuple):
    subject: str
    relation: str
    object: str


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




def _clean_list(outputs: list[Triplet]) -> list[Triplet]:
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

def make_frases(triplet:Triplet) -> str:
    return triplet.subject+ " "+triplet.relation+ " "+triplet.object

def make_frases_list(list_triplets:list[Triplet]) -> list[str]:
    return [ make_frases(triplet) for triplet in list_triplets]




@lru_cache(maxsize=1)
def _get_model():
    model_id = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoder = AutoModel.from_pretrained(model_id)
    encoder.eval()
    return tokenizer, encoder
def _compute_sentence_embeddings(list_triplets: list[Triplet]) -> Tensor:
    tokenizer, encoder = _get_model()

    texts = make_frases_list(list_triplets)
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = encoder(**inputs)

    token_embeddings = outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1)

    sentence_embedding = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    return F.normalize(sentence_embedding, p=2, dim=1)  

def create_mask(sentence_embeddings, threshold=0.85):
    similarities = sentence_embeddings @ sentence_embeddings.T

    n = similarities.shape[0]
    mask = torch.ones(n, dtype=torch.bool, device=sentence_embeddings.device)

    for i in tqdm(range(n),desc="Cleaning sinonims"):
        if not mask[i]:
            continue

        
        duplicates = similarities[i, i + 1:] > threshold
        mask[i + 1:][duplicates] = False

    return mask

def _clean(outputs: list[Triplet],threshold: float = 20) -> list[Triplet]:

    cleaned = _clean_list(outputs)

    if not cleaned:
        return []

    sentence_embeddings = _compute_sentence_embeddings(cleaned)
    mask = create_mask(sentence_embeddings, threshold)

    return [triplet for triplet, keep in zip(cleaned, mask.tolist()) if keep]


def postprocess(lists_triplets: list[list[Triplet]], threshold: float = 0.85) -> list[list[Triplet]]:
    return [_clean(triplets, threshold) for triplets in lists_triplets]

