import logging
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.postprocess import Triplet
from src.utils import load_toml_file, parse_triplet_list

logger = logging.getLogger(__name__)


def spanextraction(
    model_configs_path: str,
    dataloaders: list[DataLoader],
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 1024,
) -> list[list[Triplet]]:
    """Extrait les triplets de chaque DataLoader.

    Args:
        model_configs_path: fichier TOML du LLM (doit contenir `model_id`).
        dataloaders: DataLoaders construits par `preprocess`. Chaque batch est
            un dict {"input_ids", "attention_mask"} de forme (batch, seq_len),
            avec padding à gauche.
        tokenizer: le même tokenizer que celui utilisé dans `preprocess`
            (padding_side="left", pad_token défini).
        max_new_tokens: nombre maximal de tokens générés par document.

    Returns:
        Une liste de triplets par DataLoader, dans le même ordre.
    """
    model_config: dict[str, Any] = load_toml_file(model_configs_path)
    model_id = model_config.pop("model_id")

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_config)
    model.eval()

    results: list[list[Triplet]] = []
    with torch.no_grad():
        for i, loader in enumerate(dataloaders, start=1):
            source_triplets: list[Triplet] = []
            doc_count = 0

            for batch in tqdm(loader, desc=f"Extracting triplets from source {i}"):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                output = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

                # Padding à gauche : le prompt occupe les input_length premières
                # positions de chaque ligne, la réponse vient ensuite.
                input_length = batch["input_ids"].shape[1]
                for row in output:
                    doc_count += 1
                    answer = tokenizer.decode(
                        row[input_length:], skip_special_tokens=True
                    )

                    parsed = parse_triplet_list(answer)
                    if parsed:
                        source_triplets.extend(Triplet(*t) for t in parsed)
                    else:
                        logger.warning(
                            "No triplets parsed for source %d, document %d. Raw answer: %r",
                            i,
                            doc_count,
                            answer[:500],
                        )

            results.append(source_triplets)
    return results

    



