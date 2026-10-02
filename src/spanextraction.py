import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import load_toml_file, load_txt_file, parse_triplet_list

logger = logging.getLogger(__name__)

PROMPTS_DIR = "configs/systems_prompt"
PLACEHOLDER = "{txt}"


@dataclass
class Document:
    text: str


@dataclass
class Triplet:
    subject: str
    relation: str
    object: str


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
        # Qwen3 and similar: disable <think> so tokens aren't wasted on reasoning.
        # Templates that don't use this variable simply ignore it.
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
    """Truncate ONLY the document text so the full prompt fits in max_length,
    keeping the template and the assistant generation header intact."""
    overhead = len(
        tokenizer(
            build_prompt(tokenizer, system_prompt, user_template, ""),
            add_special_tokens=False,
        )["input_ids"]
    )
    budget = max(max_length - overhead - 8, 0)  # small safety margin
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


class SpanExtractionDataset(Dataset):
    def __init__(
        self,
        documents: list[Document],
        tokenizer: AutoTokenizer,
        system_prompt: str,
        user_template: str,
        max_length: int = 3000,
    ):
        prompts = []
        for document in documents:
            text = truncate_text_to_budget(
                tokenizer, document.text, system_prompt, user_template, max_length
            )
            prompts.append(build_prompt(tokenizer, system_prompt, user_template, text))

        # add_special_tokens=False: the chat template already includes BOS/special tokens.
        # No truncation here: the text was already trimmed, so the prompt stays intact.
        self.encodings = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.encodings.items()}


def spanextraction(
    model_configs_path: str,
    documents: list[str],
    max_length: int = 3000,
    max_new_tokens: int = 1024,
) -> list[list[Triplet]]:
    model_config: dict[str, Any] = load_toml_file(model_configs_path)

    system_prompt: str = load_txt_file(PROMPTS_DIR + "/system_prompt.txt").strip()
    user_template: str = load_txt_file(PROMPTS_DIR + "/user_template.txt").strip()

    if PLACEHOLDER not in user_template:
        raise ValueError(
            f"user_template.txt must contain the placeholder {PLACEHOLDER!r}; "
            "otherwise the model never sees the document."
        )

    model_id = model_config.pop("model_id")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # correct side for decoder-only generation

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_config)
    model.eval()

    results: list[list[Triplet]] = []
    with torch.no_grad():
        for i, document in enumerate(documents, start=1):
            dataset = SpanExtractionDataset(
                [Document(text=document)],
                tokenizer,
                system_prompt,
                user_template,
                max_length=max_length,
            )
            loader = DataLoader(dataset, batch_size=1)
            file_triplets: list[Triplet] = []

            for batch in tqdm(loader, desc=f"Extracting triplets from document {i}"):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                output = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                input_length = batch["input_ids"].shape[1]
                answer = tokenizer.decode(
                    output[0][input_length:], skip_special_tokens=True
                )

                parsed = parse_triplet_list(answer)
                if parsed:
                    file_triplets.extend(Triplet(*triplet) for triplet in parsed)
                else:
                    logger.warning(
                        "No triplets parsed for document %d. Raw answer: %r",
                        i,
                        answer[:500],
                    )

            results.append(file_triplets)
    return results

if __name__ == "__main__":
    from src.preprocess import preprocess
    print(spanextraction("configs/models/qwen2-5_config.toml", preprocess("text/test.txt", "configs/models/gliner-qwen-0.5B-v1.0.toml")))



