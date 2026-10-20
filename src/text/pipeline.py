import argparse
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.text.spanextraction import Triplet

from src.text.spanextraction import spanextraction
from src.postprocess import postprocess
from src.text.preprocess import preprocess
from src.utils import make_json_from_triplets,write_json_file
from tqdm import tqdm




def main(txt_file_path: str | list[str], 
             model_configs_path: str = "configs/models/qwen2-5_config.toml", 
             gliner_model_config_path: str = "configs/models/gliner-qwen-0.5B-v1.0.toml", 
             res_path: str | None = "res/test.json", 
             similarity_threshold: float = 0.85,
             not_filtering: bool = False, 
             stopwords: bool = False) -> None:
    
    if isinstance(txt_file_path, str):
        txt_file_paths = [txt_file_path]
    else:
        txt_file_paths = txt_file_path

    txt: list[str] = preprocess(txt_file_paths, gliner_model_config_path, not_filtering, stopwords)
    raw_concepts: list[list[Triplet]] = spanextraction(model_configs_path, txt)
    clean_concepts: list[list[Triplet]] = postprocess(raw_concepts, similarity_threshold)
    
    if clean_concepts:
        results = {txt_file_paths[i]: make_json_from_triplets(clean_concepts[i]) for i in tqdm(range(len(txt_file_paths)), desc="Making Json")}
        if res_path:
            write_json_file(res_path, results)
        else:
            print(results)
    else:
        print("Corpus None")




if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Biomedical Information Extraction Pipeline")
    parser.add_argument("data", nargs="+", help="Path to one or more input files (PDF/TXT)")
    parser.add_argument("--model_config", type=str, default="configs/models/qwen2-5_config.toml", help="Path to Qwen model config")
    parser.add_argument("--gliner_config", type=str, default="configs/models/gliner-qwen-0.5B-v1.0.toml", help="Path to GLiNER model config")
    parser.add_argument("--res_path", type=str, default="res/test.json", help="Output JSON path")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold for post-processing")
    parser.add_argument("--not_filtering", action="store_true", help="Disable filtering")
    parser.add_argument("--stopwords", action="store_true", help="Enable stopword filtering")
    
    args = parser.parse_args()
    
    main(
        txt_file_path=args.data,
        model_configs_path=args.model_config,
        gliner_model_config_path=args.gliner_config,
        res_path=args.res_path,
        similarity_threshold=args.threshold,
        not_filtering=args.not_filtering,
        stopwords=args.stopwords
    )
