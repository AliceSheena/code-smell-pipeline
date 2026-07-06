# ML Code Smell Localisation and Refactoring Pipeline

A two-stage system for Python ML code smells:
- localise smelly lines and smell type,
- generate corrected refactorings.

This repo is built for a split workflow:
- **VS Code / local**: dataset preparation, deterministic AST detection, wrapper code, training scripts, evaluation harness, and CLI scaffolding.
- **Google Colab (free T4)**: actual LoRA fine-tuning for UniXcoder and CodeT5(+).

## What is included

- `data/prepare_dataset.py` — verify JSONL entries, recover clean examples, and split the dataset.
- `data/dataset_utils.py` — build token-labelled localisation examples and seq2seq refactoring examples.
- `localisation/ast_detector.py` — deterministic AST rules for InPlace API misuse.
- `localisation/unixcoder_model.py` — UniXcoder dual-head model scaffold with LoRA support.
- `localisation/train_localisation.py` — localisation training entrypoint.
- `refactoring/codet5_model.py` — CodeT5(+) generation scaffold with training/eval loops.
- `refactoring/train_codet5.py` — refactoring training entrypoint.
- `evaluation/metrics.py` — line IoU, token F1, CodeBLEU, exact match.
- `evaluation/run_eval.py` — compare predictions and references.
- `pipeline/end_to_end.py` — inference CLI for localisation + refactoring.
- `ablation/codet5_no_prefix.py` — prefix-ablation entrypoint.
- `tests/test_smoke.py` — local smoke tests for the current build.

## Dataset format

The expected JSONL schema is:

```json
{
  "smell_type": "In-Place APIs Misused",
  "smell_location": {
    "start_line": 1,
    "end_line": 1,
    "smelly_lines": [1]
  },
  "code_smell_code": "...",
  "refactoring_code": "..."
}
```

Each line should be a valid JSON object.

## Setup

```bash
cd /Users/sheena/code-smell-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Local workflow (no GPU required)

Verify and split the dataset:

```bash
python data/prepare_dataset.py --input path/to/dataset.jsonl --output-dir ./data
```

Run the AST detector on a snippet:

```bash
python pipeline/end_to_end.py --input-file path/to/snippet.py --use-ast
```

Run smoke tests:

```bash
python3 -m unittest discover -s tests
```

## Colab workflow (GPU required)

Train localisation on Colab:

```bash
python localisation/train_localisation.py \
  --train-data data/splits/train.jsonl \
  --val-data data/splits/val.jsonl \
  --output-dir models/localisation \
  --model-name microsoft/unixcoder-base \
  --epochs 3 \
  --batch-size 2
```

Train refactoring on Colab:

```bash
python refactoring/train_codet5.py \
  --train-data data/splits/train.jsonl \
  --val-data data/splits/val.jsonl \
  --output-dir models/refactoring \
  --model-name Salesforce/codet5-base \
  --epochs 3 \
  --batch-size 2
```

## Evaluation

After training and generating predictions:

```bash
python evaluation/run_eval.py --predictions predictions.jsonl --references data/splits/test.jsonl
```

Run end-to-end inference with saved models:

```bash
python pipeline/end_to_end.py \
  --input-file path/to/snippet.py \
  --use-ast \
  --localiser-model models/localisation \
  --refactor-model models/refactoring
```

## Git status

This repository is initialized with a commit, but no remote is configured yet.

To push to GitHub, add a remote and run:

```bash
git remote add origin <your-github-url>
git push -u origin main
```

## Notes

- Training scripts are implemented but should be executed on Colab for GPU efficiency.
- The local setup is complete and verified with smoke tests.
