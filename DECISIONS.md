# Engineering Decisions

## Model architecture
- Localisation is split into a deterministic AST rule detector and a trainable UniXcoder + LoRA dual-head model.
- Refactoring uses an encoder-decoder code model (`CodeT5+` if VRAM allows, fallback to `CodeT5-base`).
- Smell-type handling is config-driven and not hardcoded, so the same architecture can absorb multiple smell categories.

## Evaluation
- Single-smell data uses binary and line-level metrics instead of macro F1.
- CodeBLEU and exact match are the primary generation metrics.
- AST validity is checked before trusting generated outputs.

## Data handling
- Dataset verification includes `ast.parse()` for every smelly code snippet.
- Group-aware splits are preferred; fallback logic should be based on actual dataset metadata.
- Duplicate and template detection is part of the data-cleaning design.

## Training design
- Localisation uses a UniXcoder dual-head model with token labels and classification loss.
- Refactoring uses CodeT5 encoder-decoder training with prefix conditioning and beam search.
- Data preparation is handled by `data/dataset_utils.py` for consistent token label mapping and example building.

## Compute constraints
- The repo targets free-tier Colab T4 limits.
- Default LoRA settings are conservative: low rank, short sequence length, batch sizes of 2–4.
- Checkpoints and resume support are required for long-running training.
- Local development is intentionally separated from GPU training: dataset prep, evaluation harness, and CLI scaffolding are built locally;
  actual UniXcoder and CodeT5(+) fine-tuning is reserved for Colab.

## Implementation priority
1. Data preparation and quality checks
2. Rule-based localisation baseline
3. Trainable localisation model scaffold
4. Refactoring model scaffold
5. Evaluation and CLI pipeline
