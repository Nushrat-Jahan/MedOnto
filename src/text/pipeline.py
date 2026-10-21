from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING

from tqdm import tqdm

from src.postprocess import postprocess
from src.text.preprocess import load_tokenizer_and_prompts, preprocess
from src.text.spanextraction import spanextraction
from src.utils import make_json_from_triplets, write_json_file

if TYPE_CHECKING:
    from src.postprocess import Triplet

logger = logging.getLogger(__name__)


def main(
    txt_file_path: str | list[str] | None = None,
    model_configs_path: str = "configs/models/qwen2-5_config.toml",
    gliner_model_config_path: str = "configs/models/gliner-qwen-0.5B-v1.0.toml",
    res_path: str | None = "res/test.json",
    similarity_threshold: float = 0.85,
    not_filtering: bool = False,
    stopwords: bool = False,
    hugging_face_dataset: str | None = None,
    column_name: str = "full_note",
    max_documents: int | None = None,
    max_length: int = 3000,
    batch_size: int = 4,
    max_new_tokens: int = 512,
) -> None:
    if txt_file_path is None and hugging_face_dataset is None:
        raise ValueError("Provide at least one input file or a Hugging Face dataset.")

    if isinstance(txt_file_path, str):
        txt_file_paths: list[str] | None = [txt_file_path]
    else:
        txt_file_paths = txt_file_path or None

    # 1. Tokenizer + prompts (partagés entre preprocess et spanextraction)
    tokenizer, system_prompt, user_template = load_tokenizer_and_prompts(model_configs_path)

    # 2. Preprocess -> DataLoaders (+ nom de la source de chacun)
    dataloaders,sources = preprocess(
        txt_file_path=txt_file_paths,
        gliner_model_path=gliner_model_config_path,
        not_filtering=not_filtering,
        stopwords=stopwords,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        user_template=user_template,
        max_length=max_length,
        hugging_face_dataset=hugging_face_dataset,
        column_name=column_name,
        max_documents=max_documents,
        batch_size=batch_size,
    )

    if not dataloaders:
        print("Corpus None: no document passed the filter.")
        return

    # 3. Extraction des triplets (un résultat par DataLoader)
    raw_concepts: list[list[Triplet]] = spanextraction(
        model_configs_path, dataloaders, tokenizer, max_new_tokens=max_new_tokens
    )

    # 4. Post-traitement
    clean_concepts: list[list[Triplet]] = postprocess(raw_concepts, similarity_threshold)

    if not clean_concepts:
        print("Corpus None")
        return

    # 5. JSON final : sources[i] <-> clean_concepts[i]
    results = {
        source: make_json_from_triplets(concepts)
        for source, concepts in tqdm(
            zip(sources, clean_concepts, strict=True),
            total=len(sources),
            desc="Making Json",
        )
    }

    if res_path:
        write_json_file(res_path, results)
    else:
        print(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Biomedical Information Extraction Pipeline")
    parser.add_argument("data", nargs="*", help="Path to zero or more input files (PDF/TXT)")
    parser.add_argument("--model_config", type=str, default="configs/models/qwen2-5_config.toml", help="Path to Qwen model config")
    parser.add_argument("--gliner_config", type=str, default="configs/models/gliner-qwen-0.5B-v1.0.toml", help="Path to GLiNER model config")
    parser.add_argument("--res_path", type=str, default="res/test.json", help="Output JSON path")
    parser.add_argument("--threshold", type=float, default=0.99,help="Similarity threshold for post-processing")
    parser.add_argument("--not_filtering", action="store_true", help="Disable filtering")
    parser.add_argument("--stopwords", action="store_true", help="Ignore stopwords when computing semantic density")
    parser.add_argument("--hf_dataset", type=str, default=None, help="Hugging Face dataset to stream (e.g. aisc-team-a1/augmented-clinical-notes)")
    parser.add_argument("--column_name", type=str, default="full_note", help="Text column of the Hugging Face dataset")
    parser.add_argument("--max_documents", type=int, default=None, help="Max documents read from the Hugging Face stream")
    parser.add_argument("--max_length", type=int, default=3000, help="Max prompt length in tokens")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate")

    args = parser.parse_args()

    if not args.data and not args.hf_dataset:
        parser.error("Provide at least one input file or --hf_dataset.")

    main(
        txt_file_path=args.data or None,
        model_configs_path=args.model_config,
        gliner_model_config_path=args.gliner_config,
        res_path=args.res_path,
        similarity_threshold=args.threshold,
        not_filtering=args.not_filtering,
        stopwords=args.stopwords,
        hugging_face_dataset=args.hf_dataset,
        column_name=args.column_name,
        max_documents=args.max_documents,
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )