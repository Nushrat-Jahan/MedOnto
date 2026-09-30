import pytest
from transformers import AutoTokenizer

from src.spanextraction import Document, SpanExtractionDataset, Triplet
from src.pipeline import main
from src.utils import load_txt_file

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
CONFIG_PATH = "configs/models/qwen2-5_config.toml"
TEST_TXT = "text/test.txt"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


def test_document_dataclass():
    doc = Document(text="hola")
    assert doc.text == "hola"


def test_triplet_dataclass():
    t = Triplet(subject="Paciente", relation="presenta", object="cefalea")
    assert t.subject == "Paciente"
    assert t.relation == "presenta"
    assert t.object == "cefalea"


def flatten_triplets(triplets):
    return " | ".join(f"{t.subject} {t.relation} {t.object}" for t in triplets)


def test_dataset_length(tokenizer):
    docs = [Document(text="texto uno"), Document(text="texto dos")]
    dataset = SpanExtractionDataset(docs, tokenizer, "sys prompt", "Context:\n{txt}")
    assert len(dataset) == 2


def test_dataset_prompt_contains_document_text(tokenizer):
    docs = [Document(text="El paciente presenta infarto agudo de miocardio.")]
    dataset = SpanExtractionDataset(docs, tokenizer, "sys prompt", "Context:\n{txt}")
    decoded = tokenizer.decode(dataset[0]["input_ids"], skip_special_tokens=True)
    assert "infarto agudo de miocardio" in decoded


def test_dataset_prompt_contains_system_prompt(tokenizer):
    docs = [Document(text="texto")]
    system_prompt = "You extract medical concepts and do not infer or add concepts."
    dataset = SpanExtractionDataset(docs, tokenizer, system_prompt, "Context:\n{txt}")
    decoded = tokenizer.decode(dataset[0]["input_ids"], skip_special_tokens=True)
    assert "do not infer or add concepts" in decoded


def test_dataset_no_txt_placeholder_left(tokenizer):
    docs = [Document(text="texto de prueba")]
    dataset = SpanExtractionDataset(docs, tokenizer, "sys", "Context:\n{txt}\nTask: extract.")
    decoded = tokenizer.decode(dataset[0]["input_ids"], skip_special_tokens=True)
    assert "{txt}" not in decoded
    assert "texto de prueba" in decoded


def test_load_txt_file_reads_utf8(tmp_path):
    p = tmp_path / "utf8.txt"
    p.write_text("Paciente con fiebre de 39 °C y dolor torácico.", encoding="utf-8")
    assert "°C" in load_txt_file(str(p))
    assert "torácico" in load_txt_file(str(p))
    assert "Ã" not in load_txt_file(str(p))


def test_main_end_to_end_single_file():
    results = main(CONFIG_PATH, TEST_TXT)
    assert isinstance(results, list)
    assert len(results) == 1
    triplets = results[0]
    assert isinstance(triplets, list)
    assert len(triplets) > 0
    assert all(isinstance(t, Triplet) for t in triplets)
    joined = flatten_triplets(triplets)
    assert "infarto" in joined
    assert "metformina" in joined
    assert "Ã" not in joined
    assert "\u00ad" not in joined


def test_main_multiple_files(tmp_path):
    second = tmp_path / "segundo.txt"
    second.write_text(
        "Paciente de 70 años con cefalea intensa y fiebre de 39 °C. "
        "Se inicia tratamiento con paracetamol.",
        encoding="utf-8",
    )
    results = main(CONFIG_PATH, [TEST_TXT, str(second)])
    assert len(results) == 2
    assert all(isinstance(r, list) and len(r) > 0 for r in results)

    first = flatten_triplets(results[0])
    assert "infarto" in first
    assert "paracetamol" not in first

    second = flatten_triplets(results[1])
    assert "paracetamol" in second
    assert "infarto" not in second
