import re
import unicodedata
from typing import Any
from gliner import GLiNER
from src.utils import load_txt_file,load_toml_file,load_pdf_file

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except Exception as e:
    raise ImportError(
        "NLTK required. Install: pip install nltk\n"
        "Then: python -c \"import nltk; nltk.download('stopwords')\""
    ) from e


_MOJIBAKE_MARKERS = ("Ã", "Â", "â€")

_LINE_BREAK_HYPHEN_RE = re.compile(r"-\n[ ]*(?=[a-záéíóúüñ])")
_SPACES_RE = re.compile(r"[ \t]+")
_LINE_EDGE_SPACES_RE = re.compile(r" ?\n ?")
_MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")

DEFAULT_LABELS = [
    "drug",
    "disease",
    "biomarker",
    "symptom"
]

THRESHOLD_DENSITY_DEFAULT = 0.05
THRESHOLD_SCORE_DEFAULT = 0.15
ENTITY_SCORE_FLOOR = 0.1

def is_pdf(path:str) -> bool: 
    return path.split(".")[-1] == "pdf" 

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

def load_gliner_model(gliner_model_path:str) -> tuple[GLiNER,list[str],float]:
    gliner_model_configs:dict[str,Any] = load_toml_file(gliner_model_path)
    model = GLiNER.from_pretrained(gliner_model_configs.get("model_id","knowledgator/gliner-qwen-0.5B-v1.0"))
    labels = gliner_model_configs.get("labels",DEFAULT_LABELS)
    threshold_score = gliner_model_configs.get("threshold_score",THRESHOLD_SCORE_DEFAULT)

    return model,labels,threshold_score

def compute_average_sematic_score(txt:str,model:GLiNER,labels:list[str])  -> float: 
    if not txt.strip():
        return 0
    entities:list[dict[str,str|float]] = model.predict_entities(txt,labels,threshold=ENTITY_SCORE_FLOOR)
    if not entities:
        return 0
    return (sum(entitie["score"] for entitie in entities)/len(entities))

def filter_text_average_score(txt: str, model: GLiNER, labels: list[str], threshold_score: float) -> bool:
    return compute_average_sematic_score(txt, model, labels) >= threshold_score


def eliminate_stopwords(txt:str) -> list[str]: 
    tokens = txt.split()
    return [
        token for token in tokens
        if token.lower() not in STOPWORDS
    ]

def compute_density_semantic(txt:str,model:GLiNER,labels:list[str],stopwords = False) -> float: 
    if not txt.strip():
        return 0
    
    if stopwords:
        n_tokens = len(eliminate_stopwords(txt))
    else:
        n_tokens = len(txt.split())
    entities:list[dict[str,str|float]] = model.predict_entities(txt,labels,threshold=ENTITY_SCORE_FLOOR)
    if not entities:
        return 0
    n_entities = len(entities)

    return n_entities/n_tokens

def filter_text_density_sematic(txt: str, model: GLiNER, labels: list[str], 
                                threshold_score: float,stopwords = False) -> bool: 
    return compute_density_semantic(txt,model,labels,stopwords) >= threshold_score
    

def filter(txt: str, model: GLiNER, labels: list[str],
           threshold_score,stopwords = False) -> bool:
    return (filter_text_average_score(txt,model,labels,threshold_score) and filter_text_density_sematic(txt,model,labels,threshold_score,stopwords))


def preprocess(txt_file_path: str | list[str],gliner_model_path:str,not_filtering:bool,stopwords:bool) -> list[str]:
    model,labels,threshold_score = load_gliner_model(gliner_model_path)
    if isinstance(txt_file_path, str):
        paths: list[str] = [txt_file_path]
    else:
        paths = txt_file_path
    potential_text:list[str] = []
    
    for path in paths:
        if is_pdf(path):
            text = load_pdf_file(path)
        else:
            text = load_txt_file(path)
        cleaned_text = clean_text(text)

        if filter(cleaned_text,model,labels,threshold_score,stopwords) or not_filtering:
            potential_text.append(cleaned_text)
    return potential_text


if __name__ == "__main__":
    print(clean_text(load_pdf_file("text/depression.pdf")))