import re
import unicodedata
from typing import Any
from gliner import GLiNER
from src.utils import load_txt_file,load_toml_file

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
THRESHOLD_SCORE_DEFAULT = 0.15
ENTITY_SCORE_FLOOR = 0.1


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

def filter_text(txt:str,model:GLiNER,labels:list[str],threshold_score:float) -> bool:
    if not txt.strip():
        return False
    entities:list[dict[str,str|float]] = model.predict_entities(txt,labels,threshold=ENTITY_SCORE_FLOOR)
    if not entities:
        return False

    return (sum(entitie["score"] for entitie in entities)/len(entities)) >= threshold_score




def preprocess(txt_file_path: str | list[str],gliner_model_path:str) -> list[str]:
    model,labels,threshold_score = load_gliner_model(gliner_model_path)
    if isinstance(txt_file_path, str):
        paths: list[str] = [txt_file_path]
    else:
        paths = txt_file_path
    potential_text:list[str] = []
    
    for path in paths:
        cleaned_text = clean_text(load_txt_file(path))
        if filter_text(cleaned_text,model,labels,threshold_score):
            potential_text.append(cleaned_text)
    return potential_text


if __name__ == "__main__":
    for cleaned in preprocess("text/test.txt","configs/models/gliner-qwen-0.5B-v1.0.toml"):
        print(cleaned)
