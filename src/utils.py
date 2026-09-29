import tomllib 

def load_txt_file(text_path:str) -> str: 
    with open(text_path) as f:
        text:str = f.read()

    return text 



def load_toml_file(toml_path:str) -> dict[str,any]: 
    with open(toml_path) as f:
        toml:dict[str,any] = tomllib.load(f)

    return toml