import json
import re
import tomllib 

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