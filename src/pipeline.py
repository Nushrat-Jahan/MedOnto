from src.preprocess import preprocess
from src.spanextraction import spanextraction, Triplet
from src.postprocess import postprocess


def main(
    model_configs_path: str,
    txt_file_path: str | list[str],
    gliner_model_path: str = "configs/models/gliner-qwen-0.5B-v1.0.toml",
) -> list[list[Triplet]]:
    clean_texts = preprocess(txt_file_path, gliner_model_path)
    lists_triplets_raw = spanextraction(model_configs_path, clean_texts)
    lists_triplets_clean = postprocess(lists_triplets_raw)
    return lists_triplets_clean


if __name__ == "__main__":
    print(main("configs/models/qwen2-5_config.toml", "text/test.txt"))
