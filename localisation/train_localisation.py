import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset_utils import load_jsonl, build_localisation_examples
from localisation.unixcoder_model import UniXcoderConfig, UniXcoderLocaliser


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the UniXcoder localisation model.")
    parser.add_argument("--train-data", required=True, help="Path to train JSONL file.")
    parser.add_argument("--val-data", required=True, help="Path to validation JSONL file.")
    parser.add_argument("--output-dir", required=True, help="Directory to save model checkpoints.")
    parser.add_argument("--model-name", default="microsoft/unixcoder-base", help="Hugging Face model name.")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    train_entries = load_jsonl(Path(args.train_data))
    val_entries = load_jsonl(Path(args.val_data))

    train_examples = build_localisation_examples(train_entries, args.model_name, args.max_length)
    val_examples = build_localisation_examples(val_entries, args.model_name, args.max_length)

    config = UniXcoderConfig(model_name=args.model_name, max_length=args.max_length)
    model = UniXcoderLocaliser(config=config)
    model.train(train_examples, val_examples, args.output_dir, epochs=args.epochs, batch_size=args.batch_size)

    print(f"Saved localisation model to {args.output_dir}")


if __name__ == "__main__":
    main()
