from unittest.mock import patch

from src.preprocess import clean_text, filter_text, load_gliner_model, preprocess

GLINER_CONFIG = "configs/models/gliner-qwen-0.5B-v1.0.toml"


class StubGlinerModel:
    def __init__(self, entities):
        self._entities = entities

    def predict_entities(self, txt, labels, threshold=0.5):
        if not txt.strip():
            return []
        return [e for e in self._entities if e["score"] >= threshold]


def test_clean_text_collapses_whitespace():
    assert clean_text("hombre   de   58   a\u00f1os") == "hombre de 58 a\u00f1os"


def test_clean_text_removes_soft_hyphen():
    assert clean_text("hipo\u00adtenso") == "hipotenso"


def test_clean_text_rejoins_line_broken_word():
    assert clean_text("hipo-\ntenso") == "hipotenso"


def test_clean_text_rejoins_line_broken_word_with_space():
    assert clean_text("hipo-\n tenso") == "hipotenso"


def test_clean_text_keeps_real_hyphen_before_capital():
    assert clean_text("alto-\nLa") == "alto-\nLa"


def test_clean_text_nfc_normalization():
    assert clean_text("e\u0301") == "\u00e9"


def test_clean_text_preserves_correct_text():
    assert clean_text("infarto agudo de miocardio") == "infarto agudo de miocardio"


def test_clean_text_fixes_mojibake():
    original = "i\u00f1farto agudo"
    mojibake = original.encode("utf-8").decode("latin-1")
    assert "\u00c3" in mojibake
    assert clean_text(mojibake) == original


def test_filter_text_no_entities_returns_false():
    model = StubGlinerModel([])
    assert filter_text("Paciente con fiebre.", model, ["symptom"], 0.6) is False


def test_filter_text_blank_text_returns_false():
    model = StubGlinerModel([{"score": 0.9}])
    assert filter_text("   \n  ", model, ["symptom"], 0.6) is False


def test_filter_text_average_score_above_threshold():
    model = StubGlinerModel([{"score": 0.7}, {"score": 0.9}])
    assert filter_text("Paciente con fiebre.", model, ["symptom"], 0.6) is True


def test_filter_text_average_score_below_threshold():
    model = StubGlinerModel([{"score": 0.5}, {"score": 0.6}])
    assert filter_text("Paciente con fiebre.", model, ["symptom"], 0.8) is False


def test_load_gliner_model_reads_config(tmp_path):
    toml_path = tmp_path / "gliner.toml"
    toml_path.write_text('model_id = "some/model"\nlabels = ["drug", "disease"]\nthreshold_score = 0.7\n', encoding="utf-8")
    with patch("src.preprocess.GLiNER.from_pretrained", return_value="model") as pretrained:
        model, labels, threshold_score = load_gliner_model(str(toml_path))
    pretrained.assert_called_once_with("some/model")
    assert model == "model"
    assert labels == ["drug", "disease"]
    assert threshold_score == 0.7


def test_preprocess_single_file(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("Paciente   con   fiebre.", encoding="utf-8")
    with patch("src.preprocess.load_gliner_model", return_value=(StubGlinerModel([{"score": 0.9}]), ["symptom"], 0.6)):
        assert preprocess(str(p), GLINER_CONFIG) == ["Paciente con fiebre."]


def test_preprocess_filters_out_documents_without_entities(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("Hoy hace un día soleado.", encoding="utf-8")
    with patch("src.preprocess.load_gliner_model", return_value=(StubGlinerModel([]), ["symptom"], 0.6)):
        assert preprocess(str(p), GLINER_CONFIG) == []


def test_preprocess_multiple_files_preserves_order(tmp_path):
    p1 = tmp_path / "a.txt"
    p1.write_text("Texto uno.", encoding="utf-8")
    p2 = tmp_path / "b.txt"
    p2.write_text("Texto dos.", encoding="utf-8")
    with patch("src.preprocess.load_gliner_model", return_value=(StubGlinerModel([{"score": 0.9}]), ["symptom"], 0.6)):
        assert preprocess([str(p1), str(p2)], GLINER_CONFIG) == ["Texto uno.", "Texto dos."]
