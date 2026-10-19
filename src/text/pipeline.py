from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.spanextraction import Triplet

from src.spanextraction import spanextraction
from src.postprocess import postprocess
from src.preprocess import preprocess
from src.utils import make_json_from_triplets,write_json_file
from tqdm import tqdm




def main(model_configs_path: str, txt_file_path: str | list[str],gliner_model_config_path:str,res_path:str,
         not_filtering:bool = False,stopwords:bool = False) -> None:
    txt: list[str] = preprocess(txt_file_path,gliner_model_config_path,not_filtering,stopwords)
    raw_concepts:list[list[Triplet]] = spanextraction(model_configs_path,txt)
    clean_concepts:list[list[Triplet]] = postprocess(raw_concepts)
    print(clean_concepts)
    if clean_concepts:
        write_json_file(res_path,{txt_file_path[i]:make_json_from_triplets(clean_concepts[i]) for i in tqdm(range(len(txt_file_path)),desc= "Making Json")})
    else:
        print("Corpus None")



if __name__ == "__main__": 
    main("configs/models/qwen2-5_config.toml", ["text/depression.pdf","text/Depression_A_Review_of_its_Definition.pdf"],"configs/models/gliner-qwen-0.5B-v1.0.toml","res/test.json", not_filtering= True) 