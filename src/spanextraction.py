from dataclasses import dataclass
from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import load_txt_file, load_toml_file, parse_concept_list

PROMPTS_DIR = "configs/systems_prompt"


@dataclass
class Document:
    text: str


class SpanExtractionDataset(Dataset):
    def __init__(
        self,
        documents: list[Document],
        tokenizer: AutoTokenizer,
        system_prompt: str,
        user_template: str,
        max_length: int = 1024,
    ):
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_template.replace("{txt}", document.text)},
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
            for document in documents
        ]
        self.encodings = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        )

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.encodings.items()}


def main(model_configs_path: str, txt_file_path: str | list[str]) -> list[list[str] | None]:
    model_config: dict[str, Any] = load_toml_file(model_configs_path)
    if isinstance(txt_file_path, str):
        txts: list[str] = [load_txt_file(txt_file_path)]
    else:
        txts: list[str] = [load_txt_file(path) for path in txt_file_path]

    system_prompt: str = load_txt_file(PROMPTS_DIR + "/system_prompt.txt").strip()
    user_template: str = load_txt_file(PROMPTS_DIR + "/user_template.txt").strip()

    model_id = model_config.pop("model_id")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    datasets = [
        SpanExtractionDataset([Document(text=txt)], tokenizer, system_prompt, user_template)
        for txt in txts
    ]
    loaders = [DataLoader(dataset, batch_size=1) for dataset in datasets]

    results: list[list[str] | None] = []
    with torch.no_grad():
        for loader in loaders:
            for batch in loader:
                batch = {key: value.to(model.device) for key, value in batch.items()}
                output = model.generate(**batch, max_new_tokens=300, do_sample=False)
                input_length = batch["input_ids"].shape[1]
                answer = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
                results.append(parse_concept_list(answer))

    return results


if __name__ == "__main__":
    print(main("configs/models/qwen2-5_config.toml", "text/test.txt"))



