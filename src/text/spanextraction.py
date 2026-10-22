import logging
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.postprocess import _to_concepts
from src.utils import load_toml_file

logger = logging.getLogger(__name__)


def spanextraction(
    model_configs_path: str,
    dataloaders: list[DataLoader],
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 512,
) -> list[list[str]]:
    """Génère, pour chaque item de chaque DataLoader, le JSON de concept clinique.

    Args:
        model_configs_path: fichier TOML du LLM (doit contenir `model_id`).
        dataloaders: DataLoaders construits par `preprocess`. Chaque batch est
            un dict {"input_ids", "attention_mask"} de forme (batch, seq_len),
            avec padding à gauche. Chaque ligne = un prompt avec UN item
            (le champ item_id doit figurer dans le Context).
        tokenizer: le même tokenizer que dans `preprocess`
            (padding_side="left", pad_token défini).
        max_new_tokens: le JSON de sortie est court, 256 suffit.

    Returns:
        Une liste de réponses brutes (str) par DataLoader, dans le même ordre.
        À passer directement à `postprocess`.
    """
    model_config: dict[str, Any] = load_toml_file(model_configs_path)
    model_id = model_config.pop("model_id")

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_config)
    model.eval()

    results: list[list[str]] = []
    with torch.no_grad():
        for i, loader in enumerate(dataloaders, start=1):
            source_outputs: list[str] = []
            doc_count = 0
            n_invalid = 0

            for batch in tqdm(loader, desc=f"Extracting concepts from source {i}"):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                output = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                input_length = batch["input_ids"].shape[1]
                for row in output:
                    doc_count += 1
                    answer = tokenizer.decode(
                        row[input_length:], skip_special_tokens=True
                    ).strip()

                    if _to_concepts(answer) is None:
                        n_invalid += 1
                        logger.warning(
                            "Invalid concept JSON for source %d, document %d. Raw answer: %r",
                            i, doc_count, answer[:500],
                        )
                        continue

                    source_outputs.append(answer)

            logger.info(
                "Source %d: %d/%d valid outputs (%d rejected)",
                i, len(source_outputs), doc_count, n_invalid,
            )
            results.append(source_outputs)
    return results