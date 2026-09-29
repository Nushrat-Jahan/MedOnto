import re 
import torch 
from typing import Any
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from src.utils import load_txt_file,load_toml_file
import re
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_SYSTEM_PROMPT = (
    "You are a biomedical information extraction assistant. "
    "Extract the medical concepts mentioned in the text: diseases, symptoms, "
    "drugs, procedures, anatomical structures and clinical findings. "
    "Rules: "
    "1) Copy each concept exactly as it appears in the text, without rewording. "
    "2) Do not add concepts that are not in the text. "
    "3) Do not explain anything. "
    "4) Return only a JSON list of strings, for example: "
    '["chest pain", "type 2 diabetes", "aspirin"]. '
    "If there are no concepts, return []."
)


class CorpusDataset(Dataset):
    def __init__(self, text: str, size: int, overlap: int):
        words: list[re.Match] = list(re.finditer(r"\S+", text))
        step: int = size - overlap
        self.items: list[dict[str, str | int]] = []
        for i in range(0, len(words), step):
            group = words[i:i + size]
            start, end = group[0].start(), group[-1].end()
            self.items.append({"text": text[start:end], "offset": start})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(tokenizer, system_prompt=DEFAULT_SYSTEM_PROMPT):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def collate(batch):
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": f"Text:\n{b['text']}\n\nTask: extract the key medical concepts."}],
                add_generation_prompt=True, tokenize=False)
            for b in batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True)
        return enc, [b["offset"] for b in batch]
    return collate


def main(model_configs_path: str, txt_file_path: str):
    model_config: dict[str, Any] = load_toml_file(model_configs_path)
    txt: str = load_txt_file(txt_file_path)

    model_id = model_config.pop("model_id")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_config)

    loader = DataLoader(
        CorpusDataset(txt, size=300, overlap=50),
        batch_size=4,
        collate_fn=make_collate(tokenizer),
    )

    answers = []
    with torch.no_grad():
        for enc, offsets in tqdm(loader, desc="Extracting"):
            enc = enc.to(model.device)
            output = model.generate(**enc, max_new_tokens=300, do_sample=False)
            generated = output[:, enc["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            answers.extend(zip(offsets, texts))

    return answers




if __name__ == "__main__": 
    print(main("configs/qwen2-5_config.toml","text/test.txt"))
