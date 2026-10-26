import logging
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from gliner import GLiNER
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoTokenizer

from src.utils import STOPWORDS, load_pdf_file, load_toml_file, load_txt_file

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
PROMPTS_DIR = "configs/systems_prompt"
PLACEHOLDER = "{txt}"

DEFAULT_LABELS = ["drug", "disease", "biomarker", "symptom"]
DEFAULT_GLINER_MODEL = "knowledgator/gliner-qwen-0.5B-v1.0"

THRESHOLD_SCORE_DEFAULT = 0.15    # score moyen minimal des entités
THRESHOLD_DENSITY_DEFAULT = 0.05  # densité sémantique minimale
ENTITY_SCORE_FLOOR = 0.1          # seuil de détection GLiNER

_MOJIBAKE_MARKERS = ("Ã", "Â", "â€")
_LINE_BREAK_HYPHEN_RE = re.compile(r"-\n[ ]*(?=[a-záéíóúüñ])")
_SPACES_RE = re.compile(r"[ \t]+")
_LINE_EDGE_SPACES_RE = re.compile(r" ?\n ?")
_MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")

Entity = dict[str, Any]


# --------------------------------------------------------------------------- #
# Nettoyage du texte
# --------------------------------------------------------------------------- #
def is_pdf(path: str) -> bool:
    return path.lower().endswith(".pdf")


def _fix_mojibake(text: str) -> str:
    if not any(marker in text for marker in _MOJIBAKE_MARKERS):
        return text
    for encoding in ("cp1252", "latin-1"):
        try:
            return text.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return text


def _clean_hyphens(text: str) -> str:
    text = text.replace("\u00ad", "")
    return _LINE_BREAK_HYPHEN_RE.sub("", text)


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES_RE.sub(" ", text)
    text = _LINE_EDGE_SPACES_RE.sub("\n", text)
    text = _MULTIPLE_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    text = _fix_mojibake(text)
    text = unicodedata.normalize("NFC", text)
    text = _clean_hyphens(text)
    text = _normalize_whitespace(text)
    return text


# --------------------------------------------------------------------------- #
# Filtre sémantique (GLiNER)
# --------------------------------------------------------------------------- #
def predict_entities(
    txt: str,
    model: GLiNER,
    labels: list[str],
    threshold: float = ENTITY_SCORE_FLOOR,
) -> list[Entity]:
    return model.predict_entities(txt, labels, threshold=threshold)


def eliminate_stopwords(txt: str) -> list[str]:
    return [token for token in txt.split() if token.lower() not in STOPWORDS]


def count_words(txt: str, ignore_stopwords: bool) -> int:
    """Nombre de mots de `txt`, en ignorant (ou non) les stopwords."""
    return len(eliminate_stopwords(txt)) if ignore_stopwords else len(txt.split())


def compute_average_semantic_score(entities: list[Entity]) -> float:
    if not entities:
        return 0.0
    return sum(entity["score"] for entity in entities) / len(entities)


def compute_density_semantic(
    entities: list[Entity],
    txt: str,
    ignore_stopwords: bool = False,
) -> float:
    """Proportion de mots du texte couverts par des entités.

    `ignore_stopwords` s'applique de la même façon au numérateur et au
    dénominateur.
    """
    n_tokens = count_words(txt, ignore_stopwords)
    if not entities or n_tokens == 0:
        return 0.0
    n_entity_tokens = sum(
        count_words(txt[entity["start"]:entity["end"]], ignore_stopwords)
        for entity in entities
    )
    return n_entity_tokens / n_tokens


@dataclass
class TextFilter:
    """Décide si un texte est assez riche en entités pour être gardé."""

    model: GLiNER
    labels: list[str]
    threshold_score: float = THRESHOLD_SCORE_DEFAULT
    threshold_density: float = THRESHOLD_DENSITY_DEFAULT
    ignore_stopwords: bool = False

    def __call__(self, text: str) -> bool:
        entities = predict_entities(text, self.model, self.labels)
        score = compute_average_semantic_score(entities)
        density = compute_density_semantic(entities, text, self.ignore_stopwords)
        keep = score >= self.threshold_score and density >= self.threshold_density
        logger.debug("score=%.3f density=%.3f keep=%s", score, density, keep)
        return keep


def load_text_filter(gliner_model_path: str, ignore_stopwords: bool = False) -> TextFilter:
    config: dict[str, Any] = load_toml_file(gliner_model_path)
    model = GLiNER.from_pretrained(config.get("model_id", DEFAULT_GLINER_MODEL))
    return TextFilter(
        model=model,
        labels=config.get("labels", DEFAULT_LABELS),
        threshold_score=config.get("threshold_score", THRESHOLD_SCORE_DEFAULT),
        threshold_density=config.get("threshold_density", THRESHOLD_DENSITY_DEFAULT),
        ignore_stopwords=ignore_stopwords,
    )


# --------------------------------------------------------------------------- #
# Prompts et tokenizer
# --------------------------------------------------------------------------- #
@dataclass
class Document:
    text: str


def load_tokenizer_and_prompts(model_configs_path: str) -> tuple[AutoTokenizer, str, str]:
    model_id = load_toml_file(model_configs_path)["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    system_prompt = load_txt_file(PROMPTS_DIR + "/system_prompt.txt").strip()
    user_template = load_txt_file(PROMPTS_DIR + "/user_template.txt").strip()
    if PLACEHOLDER not in user_template:
        raise ValueError(
            f"user_template.txt must contain the placeholder {PLACEHOLDER!r}; "
            "otherwise the model never sees the document."
        )
    return tokenizer, system_prompt, user_template


def build_prompt(
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_template: str,
    text: str,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_template.replace(PLACEHOLDER, text)},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )


def truncate_text_to_budget(
    tokenizer: AutoTokenizer,
    text: str,
    system_prompt: str,
    user_template: str,
    max_length: int,
) -> str:
    """Tronque UNIQUEMENT le texte du document pour que le prompt complet tienne
    dans max_length, en gardant intact le template et l'en-tête de génération."""
    overhead = len(
        tokenizer(
            build_prompt(tokenizer, system_prompt, user_template, ""),
            add_special_tokens=False,
        )["input_ids"]
    )
    budget = max(max_length - overhead - 8, 0)
    text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(text_ids) <= budget:
        return text
    logger.warning(
        "Document truncated from %d to %d tokens (max_length=%d).",
        len(text_ids),
        budget,
        max_length,
    )
    return tokenizer.decode(text_ids[:budget], skip_special_tokens=True)


def encode_document(
    tokenizer: AutoTokenizer,
    text: str,
    system_prompt: str,
    user_template: str,
    max_length: int,
) -> dict[str, torch.Tensor]:
    """Texte -> tenseurs 1D (seq_len,) ; le DataLoader ajoute la dimension batch."""
    text = truncate_text_to_budget(tokenizer, text, system_prompt, user_template, max_length)
    prompt = build_prompt(tokenizer, system_prompt, user_template, text)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    return {key: value.squeeze(0) for key, value in encoded.items()}


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class SpanExtractionDataset(Dataset):
    """Dataset en mémoire (fichiers locaux déjà lus et nettoyés)."""

    def __init__(
        self,
        documents: list[Document],
        tokenizer: AutoTokenizer,
        system_prompt: str,
        user_template: str,
        max_length: int = 3000,
    ):
        self.documents = documents
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return encode_document(
            self.tokenizer,
            self.documents[index].text,
            self.system_prompt,
            self.user_template,
            self.max_length,
        )


class SpanExtractionDatasetHuggingFace(IterableDataset):
    """Dataset en streaming depuis le Hub Hugging Face."""

    def __init__(
        self,
        hugging_face_dataset: str,
        column_name: str,
        tokenizer: AutoTokenizer,
        system_prompt: str,
        user_template: str,
        text_filter: TextFilter | None = None,
        max_length: int = 3000,
        split: str = "train",
        max_documents: int | None = None,
    ):
        stream = load_dataset(hugging_face_dataset, split=split, streaming=True)
        self.stream = stream.take(max_documents) if max_documents else stream
        self.column_name = column_name
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.text_filter = text_filter
        self.max_length = max_length

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for row in self.stream:
            text = clean_text(row[self.column_name])
            if self.text_filter is not None and not self.text_filter(text):
                continue
            yield encode_document(
                self.tokenizer,
                text,
                self.system_prompt,
                self.user_template,
                self.max_length,
            )


# --------------------------------------------------------------------------- #
# Preprocess -> DataLoaders, Sources
# --------------------------------------------------------------------------- #
def preprocess(
    txt_file_path: str | list[str] | None,
    gliner_model_path: str,
    not_filtering: bool,
    stopwords: bool,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_template: str,
    max_length: int = 3000,
    hugging_face_dataset: str | None = None,
    column_name: str = "full_note",
    split: str = "train",
    max_documents: int | None = None,
    batch_size: int = 1,
) -> tuple[list[DataLoader], list[str]]:
    """Construit les DataLoaders prêts pour `spanextraction`.

    Renvoie `(dataloaders, sources)` : `sources[i]` est le chemin du fichier
    (ou le nom du dataset HF) qui a produit `dataloaders[i]`. Les fichiers
    écartés par le filtre n'y figurent pas.

    `stopwords=True` : les stopwords sont ignorés dans le calcul de la densité.
    """
    text_filter = None if not_filtering else load_text_filter(gliner_model_path, stopwords)

    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return tokenizer.pad(batch, return_tensors="pt")

    dataloaders: list[DataLoader] = []
    sources: list[str] = []

    # --- Fichiers locaux ---
    if txt_file_path is not None:
        paths = [txt_file_path] if isinstance(txt_file_path, str) else txt_file_path
        for path in paths:
            raw = load_pdf_file(path) if is_pdf(path) else load_txt_file(path)
            text = clean_text(raw)

            if text_filter is not None and not text_filter(text):
                logger.info("Document %s écarté par le filtre.", path)
                continue

            dataset = SpanExtractionDataset(
                [Document(text=text)],
                tokenizer,
                system_prompt,
                user_template,
                max_length=max_length,
            )
            dataloaders.append(
                DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
            )
            sources.append(path)

    # --- Dataset Hugging Face en streaming ---
    if hugging_face_dataset is not None:
        dataset = SpanExtractionDatasetHuggingFace(
            hugging_face_dataset=hugging_face_dataset,
            column_name=column_name,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_template=user_template,
            text_filter=text_filter,
            max_length=max_length,
            split=split,
            max_documents=max_documents,
        )
        # num_workers=0 : le filtre GLiNER vit dans __iter__, des workers
        # dupliqueraient les documents (et le modèle).
        dataloaders.append(
            DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)
        )
        sources.append(hugging_face_dataset)

    return dataloaders, sources

