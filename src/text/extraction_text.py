from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from tqdm import tqdm

from src.postprocess import postprocess
from src.text.preprocess import load_tokenizer_and_prompts, preprocess
from src.text.spanextraction import spanextraction
from src.utils import make_json_from_concepts, write_json_file

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
    dataloaders, sources = preprocess(
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

    # 3. Extraction : chaque document est envoyé à Qwen via Ollama (même modèle que les autres parties)
    raw_outputs = spanextraction(
        dataloaders, system_prompt, user_template, max_new_tokens=max_new_tokens
    )

    clean_concepts = postprocess(raw_outputs, similarity_threshold)

    n_concepts = sum(len(c) for c in clean_concepts)
    if n_concepts == 0:
        print("Corpus None: no valid concept extracted.")
        return
    logger.info("%d concepts kept across %d sources", n_concepts, len(sources))

    
    results = {
        source: make_json_from_concepts(concepts)
        for source, concepts in tqdm(
            zip(sources, clean_concepts, strict=True),
            total=len(sources),
            desc="Making Json",
        )
    }

    if res_path:
        write_json_file(res_path, results)
        logger.info("Results written to %s", res_path)
    else:
        print(results)


def run_text(text_path: str | Path) -> Path | None:
    """Extraction d'un texte (PDF ou TXT) : du fichier au fichier JSON commun.

    Appelable par un autre programme (comme src/main.py). Lance main() ci-dessus sur un seul
    fichier, avec le même seuil de doublons que la commande (0.99), et range le résultat dans
    outputs/<nom_fichier>_<id>/concepts_<id>.json, comme les autres parties.

    Paramètre :
        text_path : le fichier texte, ex. "data/texte/note.pdf".

    Renvoie :
        le chemin du fichier JSON commun écrit (c'est l'entrée de la vérification),
        ou None si aucun concept n'a été extrait (main() n'écrit alors aucun fichier).

    Lève :
        FileNotFoundError : si le fichier n'existe pas.
    """
    text_path = Path(text_path)
    if not text_path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {text_path}")

    # Dossier : nom du fichier + empreinte de son contenu ; fichier : empreinte des prompts
    file_id = hashlib.sha1(text_path.read_bytes()).hexdigest()[:8]
    out_dir = Path("outputs") / f"{text_path.stem.replace(' ', '_')}_{file_id}"
    concepts_path = out_dir / f"concepts_{prompts_id()}.json"

    main(txt_file_path=str(text_path), res_path=str(concepts_path), similarity_threshold=0.99)

    if not concepts_path.is_file():
        return None
    return concepts_path


def run_text_hf(dataset_name: str, column_name: str, max_documents: int) -> Path | None:
    """Extraction d'un dataset Hugging Face de textes : les max_documents premiers textes → JSON commun.

    Les textes sont lus en direct sur Hugging Face (streaming), sans télécharger le dataset.
    Appelable par un autre programme (comme src/main.py). Lance main() ci-dessus, avec le même
    seuil de doublons que la commande (0.99), et range le résultat dans outputs/, comme les autres parties.

    Paramètres :
        dataset_name  : le nom du dataset sur Hugging Face, ex. "aisc-team-a1/augmented-clinical-notes".
        column_name   : la colonne qui contient le texte, ex. "full_note".
        max_documents : le nombre de textes lus (chaque texte demande un appel à Qwen).

    Renvoie :
        le chemin du fichier JSON commun écrit (c'est l'entrée de la vérification),
        ou None si aucun concept n'a été extrait.
    """
    # Dossier : nom du dataset, colonne et nombre de textes ; fichier : empreinte des prompts
    folder = dataset_name.replace("/", "_") + f"_{column_name}_{max_documents}docs"
    concepts_path = Path("outputs") / folder / f"concepts_{prompts_id()}.json"

    main(
        hugging_face_dataset=dataset_name,
        column_name=column_name,
        max_documents=max_documents,
        res_path=str(concepts_path),
        similarity_threshold=0.99,
    )

    if not concepts_path.is_file():
        return None
    return concepts_path


def prompts_id() -> str:
    """Empreinte courte du prompt système et du user_template : elle change si l'un des deux change."""
    prompts = Path("configs/systems_prompt/system_prompt.txt").read_text(encoding="utf-8")
    prompts = prompts + Path("configs/systems_prompt/user_template.txt").read_text(encoding="utf-8")
    return hashlib.sha1(prompts.encode("utf-8")).hexdigest()[:8]


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