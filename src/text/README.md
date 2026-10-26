# `src/text/` — medical and mental-health text track

This package extracts **clinical concepts** from unstructured medical or mental-health
documents (PDF / TXT / Hugging Face stream) with an LLM, normalizes them, and stores them in
a shared JSON format ready for SNOMED CT verification and ontology building.

It is the counterpart of `src/bdd/` (tabular cohort data) and `src/questions/` (questionnaire
items) in the common architecture described in the [root README](../../README.md).

## Package layout

```text
src/text/
├── __init__.py        ← loads .env, silences noisy warnings (FutureWarning, hf_hub, tokenizer)
├── pipeline.py        ← entry point: main() + argparse CLI (orchestrates the 5 stages)
├── preprocess.py      ← text cleaning, GLiNER semantic filter, prompt building, DataLoaders
└── spanextraction.py  ← batched LLM generation of the concept JSON
```

All modules are imported from the **project root**, e.g.
`from src.text.pipeline import main`.

## Pipeline overview

`src/text/pipeline.py` orchestrates five stages:

```text
documents (PDF/TXT/HF stream)
   │
   │  1. load_tokenizer_and_prompts()   → tokenizer + system_prompt + user_template
   ▼
preprocess()  ──►  clean_text()          → mojibake fix, NFC, hyphen repair, whitespace
                 └─►  TextFilter (GLiNER) → keep only entity-rich documents
                 └─►  span filtering       → keep sentences containing detected entities
                 └─►  build_prompt()      → chat template with the {txt} placeholder
                 └─►  DataLoader[]        → one loader per local file, one for the HF stream
   │
   │  2. spanextraction()  → model.generate() per batch, invalid JSON rejected
   ▼
postprocess()  ──►  normalization, junk removal, exact dedup
                └─►  Bio_ClinicalBERT embeddings → near-duplicate removal (cosine threshold)
   │
   │  3. make_json_from_concepts()  → list[dict] per source
   ▼
write_json_file(res_path, results)   → { source: [concept, ...] }
```

## Requirements

The package needs the following packages (a local virtual environment with them installed lives
in `myenv/`, which is git-ignored and therefore not part of the repository):

| Package | Used for |
|---|---|
| `torch` | tensor ops, `DataLoader`, generation |
| `transformers` | `AutoTokenizer`, `AutoModelForCausalLM` |
| `gliner` | semantic filter (`knowledgator/gliner-qwen-0.5B-v1.0`) |
| `datasets` | streaming a Hugging Face dataset |
| `pypdf` | reading PDF documents |
| `python-dotenv` | loading `.env` (imported by `src/text/__init__.py`) |
| `tqdm` | progress bars |
| `nltk` | English stopwords, used by the density filter and `postprocess` |

The following models are downloaded from the Hugging Face Hub on first run, so **network access
is required**:

- the LLM of `configs/models/qwen2-5_config.toml` (default `Qwen/Qwen2.5-1.5B-Instruct`),
- the filter model of `configs/models/gliner-qwen-0.5B-v1.0.toml` (only when the filter is on),
- `emilyalsentzer/Bio_ClinicalBERT`, loaded by `src/postprocess.py` for the similarity dedup.

`nltk` needs its `stopwords` corpus (`python -m nltk.downloader stopwords`); if it is missing the
code still runs, but the semantic-density filter and the stopword cleanup are degraded.

## Configuration files

| File | Role |
|---|---|
| `configs/models/qwen2-5_config.toml` | LLM: `model_id` plus any `from_pretrained` kwarg (`dtype`, `device_map`, `attn_implementation`). `model_id` is popped, the remaining keys are forwarded to `AutoModelForCausalLM`. |
| `configs/models/gliner-qwen-0.5B-v1.0.toml` | Filter: `model_id`, `labels`, `threshold_score`, `threshold_density`. |
| `configs/systems_prompt/system_prompt.txt` | System prompt. |
| `configs/systems_prompt/user_template.txt` | User prompt. **Must contain the `{txt}` placeholder**, otherwise `load_tokenizer_and_prompts` raises a `ValueError`. |

The prompt asks the model for a strict JSON object per document, based only on the supplied
medical context (terminology matching is not part of the extraction step):

```json
{
  "item_id": "...",
  "concept_proposed": "...",
  "category": "finding | symptom | disorder | behavior | observable_entity | assessment_scale",
  "relation": { "predicate": "...", "target_concept": "..." } or null,
  "evidence": "...",
  "synonyms": ["...", "..."]
}
```

## Running the pipeline

### Command line

Run it **from the project root** with the module form, so that the absolute imports
(`from src.postprocess import ...`) resolve:

```bash
python -m src.text.pipeline data/depression.pdf
```

Several documents at once, with a custom output file:

```bash
python -m src.text.pipeline data/depression.pdf "data/Depression_A_Review_of_its_Definition.pdf" \
    --res_path res/run1.json --batch_size 8 --max_new_tokens 512
```

Stream a Hugging Face dataset instead of local files:

```bash
python -m src.text.pipeline --hf_dataset aisc-team-a1/augmented-clinical-notes \
    --column_name full_note --max_documents 100 --res_path res/hf_run.json
```

Disable the semantic filter (keep every document, whatever its content):

```bash
python -m src.text.pipeline data/depression.pdf --not_filtering
```

> `python src/text/pipeline.py` does **not** work: running the file directly puts `src/text/`
> on `sys.path` instead of the project root, and the `src.*` imports fail.

### Python API

`main()` is the same entry point used by the CLI. It returns `None` and writes the results to
`res_path`; pass `res_path=None` to print them on stdout instead.

```python
from src.text.pipeline import main

main(
    txt_file_path="data/depression.pdf",   # or a list of paths, or None with a HF dataset
    model_configs_path="configs/models/qwen2-5_config.toml",
    gliner_model_config_path="configs/models/gliner-qwen-0.5B-v1.0.toml",
    res_path="res/test.json",
    similarity_threshold=0.85,
)
```

With a Hugging Face dataset:

```python
main(
    txt_file_path=None,
    hugging_face_dataset="aisc-team-a1/augmented-clinical-notes",
    column_name="full_note",
    max_documents=100,
    res_path="res/hf_run.json",
)
```

### `main()` parameters

| Parameter | Default | Description |
|---|---|---|
| `txt_file_path` | `None` | Path, or list of paths, of the input documents (`.pdf` or `.txt`). |
| `model_configs_path` | `configs/models/qwen2-5_config.toml` | TOML config of the LLM. |
| `gliner_model_config_path` | `configs/models/gliner-qwen-0.5B-v1.0.toml` | TOML config of the filter model. |
| `res_path` | `res/test.json` | Output JSON file. `None` prints the result instead of writing it. |
| `similarity_threshold` | `0.85` | Cosine similarity above which two concepts are considered duplicates. |
| `not_filtering` | `False` | Skip the GLiNER semantic filter entirely. |
| `stopwords` | `False` | Ignore stopwords when computing the semantic density. |
| `hugging_face_dataset` | `None` | Dataset id to stream from the Hub. |
| `column_name` | `full_note` | Name of the text column in the Hugging Face dataset. |
| `max_documents` | `None` | Maximum number of documents read from the stream. |
| `max_length` | `3000` | Maximum prompt length in tokens. |
| `batch_size` | `4` | Batch size for generation. |
| `max_new_tokens` | `512` | Maximum number of tokens generated per document. |

`main()` raises `ValueError` when neither `txt_file_path` nor `hugging_face_dataset` is given.

### CLI flags

| Flag | Equivalent argument | Default |
|---|---|---|
| `data` (positional, repeatable) | `txt_file_path` | — |
| `--model_config` | `model_configs_path` | `configs/models/qwen2-5_config.toml` |
| `--gliner_config` | `gliner_model_config_path` | `configs/models/gliner-qwen-0.5B-v1.0.toml` |
| `--res_path` | `res_path` | `res/test.json` |
| `--threshold` | `similarity_threshold` | **`0.99`** (see the note below) |
| `--not_filtering` | `not_filtering` | off |
| `--stopwords` | `stopwords` | off |
| `--hf_dataset` | `hugging_face_dataset` | `None` |
| `--column_name` | `column_name` | `full_note` |
| `--max_documents` | `max_documents` | `None` |
| `--max_length` | `max_length` | `3000` |
| `--batch_size` | `batch_size` | `4` |
| `--max_new_tokens` | `max_new_tokens` | `512` |

> **Note:** the CLI default of `--threshold` is `0.99`, while the `main()` default is `0.85`.
> The CLI is therefore *more* aggressive when removing near-duplicate concepts. Pass
> `--threshold 0.85` explicitly to reproduce the API behaviour.

## What each stage does

### 1. Tokenizer and prompts — `load_tokenizer_and_prompts()`

Loads the tokenizer of the LLM (padding on the **left**, `pad_token` set to `eos_token` when
missing) and the two prompt files. The `{txt}` placeholder of `user_template.txt` is replaced
by the document, and the messages are rendered with `tokenizer.apply_chat_template()`.

### 2. Preprocessing — `preprocess()`

- **Cleaning** (`clean_text`): repairs mojibake (`Ã`, `Â`, `â€`), applies Unicode NFC
  normalization, removes soft hyphens, rejoins words broken by a hyphen at the end of a line
  (`hipo-\ntenso` → `hipotenso`) and collapses redundant whitespace.
- **Semantic filter** (`TextFilter`): GLiNER detects entities matching the configured labels;
  a document is kept when the mean entity score is `>= threshold_score` **and** the semantic
  density (share of the words covered by an entity) is `>= threshold_density`. With
  `stopwords=True`, stopwords are removed from both the numerator and the denominator of the
  density. Filtered-out documents are logged and never reach the LLM.
- **Span filtering** (`filter_text_by_spans`): the entities already detected by GLiNER during
  document filtering are reused to keep only the sentences/lines that contain a detected span.
  The LLM therefore receives a shorter, entity-focused document, which speeds up generation
  without running GLiNER a second time. This step is skipped when `not_filtering=True`.
- **Truncation** (`truncate_text_to_budget`): only the *document text* is truncated so that the
  whole prompt fits in `max_length` tokens; the template and the generation header stay intact.
- **Batching**: each local file becomes its own `DataLoader` containing one document, so the
  source of every output is preserved. The Hugging Face stream becomes a single `DataLoader`
  over an `IterableDataset` (`num_workers=0`, otherwise the filter would be duplicated across
  workers).

`preprocess()` returns `(dataloaders, sources)`, where `sources[i]` is the path of the file
(or the dataset name) that produced `dataloaders[i]`.

### 3. Generation — `spanextraction()`

Loads the LLM from the TOML config, then generates for each batch with `do_sample=False`
(greedy). Only the newly generated tokens are decoded. An answer whose JSON cannot be parsed
into valid concepts is **rejected** and logged with its first 500 characters; the valid
answers of the same source are kept.

### 4. Post-processing — `postprocess()`

Shared module `src/postprocess.py`:

- lowercases and normalizes whitespace, removes punctuation, stopwords, anonymized tokens
  (`___`), digit-only strings and lab-value patterns;
- drops exact duplicates on the `(concept_proposed, category)` pair, keeping the first
  occurrence;
- embeds the remaining concepts with `emilyalsentzer/Bio_ClinicalBERT` and removes concepts
  whose cosine similarity is **above** `similarity_threshold`.

### 5. Output — `make_json_from_concepts()` and `write_json_file()`

The result is a JSON object mapping each source to its list of concepts. Real output for a
single document:

```json
{
  "data/depression.pdf": [
    {
      "item_id": "chest_pain_and_right_back_swelling",
      "concept_proposed": "chest pain",
      "category": "finding",
      "relation": {
        "predicate": "measures",
        "target_concept": "right back swelling"
      },
      "evidence": "The patient reported chest pain and swelling in the right back area, indicating these symptoms.",
      "synonyms": ["thoracic pain", "pain in chest"]
    }
  ]
}
```

The key of each entry is the source path (or the dataset name for a Hugging Face run).
`relation` is `null` when the model did not provide a valid predicate/target pair.
`synonyms` contains a list of alternative terms for `concept_proposed` (empty array if none).

### 6. Downstream verification — `src/verification.py`

The JSON output is consumed by the shared verification pipeline, which applies a three-step
fallback for every concept:

1. **Direct SNOMED lookup** of `concept_proposed`.
2. **Synonym lookup** using the `synonyms` list. When a synonym matches, the children of the
   matched SNOMED concept are compared with the original concept using embeddings; the most
   relevant child is kept when it is closer than the synonym.
3. **Embedding fallback** on the original concept when no synonym matches.

The verifier returns `concept`, `in_snomed`, and `verification_step`, then aggregates the results
into a pandas DataFrame. See the [root README](../../README.md#verification-pipeline-srcverificationpy)
for the full workflow and output columns.

## Behaviour and gotchas

- **Empty results are reported, not raised.** If the filter removes every document the pipeline
  prints `Corpus None: no document passed the filter.`; if no concept survives the
  post-processing it prints `Corpus None: no valid concept extracted.` In both cases nothing is
  written to `res_path`.
- **`batch_size` only matters for the Hugging Face stream.** Local files are loaded one by one,
  each in its own loader.
- **The Hugging Face split is fixed to `train`.** `preprocess()` accepts a `split` argument, but
  `main()` and the CLI do not expose it.
- **Prompts must be adapted to the input type.** The prompts currently in `configs/systems_prompt/`
  target questionnaire items, while the package also handles full clinical notes.
- **The models are downloaded on first run** and cached in the usual Hugging Face cache.
- **The `UNEXPECTED` keys in the `BertModel LOAD REPORT` are harmless.** `postprocess` loads
  `emilyalsentzer/Bio_ClinicalBERT` as a plain encoder, so the pretrained
  `cls.predictions.*` / `cls.seq_relationship.*` heads are reported as unexpected and ignored.
  The run is correct despite the report.
- **Two progress bars are expected**: `Extracting concepts from source N` (generation) and
  `Cleaning synonyms` (similarity dedup in `postprocess`), plus `Making Json` when writing the
  result.
