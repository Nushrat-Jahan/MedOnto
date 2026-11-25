import logging

import ollama
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.postprocess import _to_concepts
from src.text.preprocess import PLACEHOLDER

logger = logging.getLogger(__name__)

# Le même modèle et les mêmes réglages que les autres parties (bdd, questionnaires, vérification),
# pour que les résultats des trois parties soient comparables.
MODEL = "qwen2.5:7b"  # Qwen 2.5 7B, en local via Ollama


def spanextraction(
    dataloaders: list[DataLoader],
    system_prompt: str,
    user_template: str,
    max_new_tokens: int = 512,
) -> list[list[str]]:
    """Génère, pour chaque document de chaque DataLoader, le JSON de concept clinique.

    Chaque document est envoyé à Qwen via Ollama (le logiciel Ollama doit être lancé), avec le
    prompt système commun et le user_template, comme dans les autres parties : temperature=0
    (même question -> même réponse) et réponse obligatoirement en JSON.

    Args:
        dataloaders: DataLoaders construits par `preprocess`. Chaque batch est une liste de
            textes (un texte = un document, déjà coupé pour tenir dans le prompt).
        system_prompt: le prompt système commun (configs/systems_prompt/system_prompt.txt).
        user_template: le message propre à la partie texte, avec le repère {txt}.
        max_new_tokens: longueur maximum de la réponse (le JSON de sortie est court).

    Returns:
        Une liste de réponses brutes (str) par DataLoader, dans le même ordre.
        À passer directement à `postprocess`.
    """
    results: list[list[str]] = []
    for i, loader in enumerate(dataloaders, start=1):
        source_outputs: list[str] = []
        doc_count = 0
        n_invalid = 0

        for texts in tqdm(loader, desc=f"Extracting concepts from source {i}"):
            for text in texts:
                doc_count += 1
                response = ollama.chat(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_template.replace(PLACEHOLDER, text)},
                    ],
                    options={"temperature": 0, "num_predict": max_new_tokens},
                    format="json",
                )
                answer = response["message"]["content"].strip()

                if not _to_concepts(answer):
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
