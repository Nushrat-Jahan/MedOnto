from src.spanextraction import spanextraction, Triplet
from src.preprocess import clean_outputs
from tqdm import tqdm



def main(model_configs_path: str, txt_file_path: str | list[str]):
    lists_triplets_raw:list[list[Triplet]] = spanextraction(model_configs_path,txt_file_path)
    lists_triplets_clean:list[list[Triplet]] = []
    for list_triplets in tqdm(lists_triplets_raw,desc = "Cleaning Triplets"):
        lists_triplets_raw.append(clean_outputs(list_triplets))

    

    print(lists_triplets_clean)









if __name__ == "__main__": 
    main("configs/models/qwen2-5_config.toml", "text/test.txt")