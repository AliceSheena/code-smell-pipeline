# ML Code Smell Localisation and Refactoring Pipeline

This repository implements a two-stage pipeline for Python ML code smell localisation and refactoring.
It is designed to handle multiple smell types and scale from the In-Place API Misuse subset to a broader ML code smell taxonomy.

## Project structure

- `data/prepare_dataset.py` — load, verify, and split the dataset.
- `localisation/ast_detector.py` — deterministic AST-based InPlace API misuse detector.
- `localisation/unixcoder_model.py` — UniXcoder + LoRA localisation scaffold.
- `refactoring/codet5_model.py` — CodeT5(+) refactoring scaffold.
- `refactoring/train_codet5.py` — entry point for CodeT5 training.
- `baselines/few_shot_baseline.py` — non-fine-tuned few-shot comparison scaffold.
- `evaluation/metrics.py` — line IoU, token F1, CodeBLEU, exact match.
- `evaluation/run_eval.py` — evaluation runner for predictions and references.
- `pipeline/end_to_end.py` — single-file CLI for localisation + refactoring.
- `ablation/codet5_no_prefix.py` — prefix-ablation setup.
- `requirements.txt` — pinned open-source dependencies.
- `DECISIONS.md` — high-level decisions and design rationale.

## Setup

1. Create a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Prepare your dataset using the expected JSONL schema.

```bash
python data/prepare_dataset.py --input path/to/dataset.jsonl --output-dir ./data
```

## Notes

- The AST detector is a fast rule-based baseline for InPlace API misuse, with explicit line extraction.
- The localisation and refactoring models now include training loops using Hugging Face models and tokenizer-based data preparation.
- The code is written to support multi-smell datasets once data grouping and labels are available.

## Running the pipeline

This project is split across two environments:

1. **VS Code (local)**: build, validate, and prepare everything that does not require a GPU.
2. **Google Colab (free T4)**: execute the actual LoRA training runs for UniXcoder and CodeT5(+).

### Local workflow

Prepare the dataset:

```bash
python data/prepare_dataset.py --input path/to/dataset.jsonl --output-dir ./data
```

Run the rule-based AST localisation on a snippet:

```bash
python pipeline/end_to_end.py --input-file path/to/snippet.py --use-ast
```

### Colab training workflow

Train the localisation model on Colab:

```bash
python localisation/train_localisation.py \
  --train-data data/splits/train.jsonl \
  --val-data data/splits/val.jsonl \
  --output-dir models/localisation \
  --model-name microsoft/unixcoder-base \
  --epochs 3 \
  --batch-size 2
```

Train the refactoring model on Colab:

```bash
python refactoring/train_codet5.py \
  --train-data data/splits/train.jsonl \
  --val-data data/splits/val.jsonl \
  --output-dir models/refactoring \
  --model-name Salesforce/codet5-base \
  --epochs 3 \
  --batch-size 2
```

### Evaluation and inference

After Colab training, evaluate predictions locally:

```bash
python evaluation/run_eval.py --predictions predictions.jsonl --references data/splits/test.jsonl
```

Run end-to-end inference with the saved models:

```bash
python pipeline/end_to_end.py --input-file path/to/snippet.py --use-ast --localiser-model models/localisation --refactor-model models/refactoring
```

## Colab fit

The repo is designed to run the actual model fine-tuning on a free-tier Google Colab T4 GPU, while keeping all development and scaffold verification local.
