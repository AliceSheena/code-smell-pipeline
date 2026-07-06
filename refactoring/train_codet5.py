import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset_utils import load_jsonl, build_refactoring_examples
from refactoring.codet5_model import CodeT5Config, CodeT5Refactor


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CodeT5-based refactoring model.")
    parser.add_argument("--train-data", required=True, help="Path to training JSONL file.")
    parser.add_argument("--val-data", required=True, help="Path to validation JSONL file.")
    parser.add_argument("--output-dir", required=True, help="Directory to save checkpoints.")
    parser.add_argument("--model-name", default="Salesforce/codet5-base", help="Hugging Face model name.")
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--no-prefix", action="store_true", help="Do not use the smell-type prefix input conditioning.")
    args = parser.parse_args()

    train_entries = load_jsonl(Path(args.train_data))
    val_entries = load_jsonl(Path(args.val_data))

    train_examples = build_refactoring_examples(
        train_entries,
        args.model_name,
        args.max_source_length,
        args.max_target_length,
        prefix=not args.no_prefix,
    )
    val_examples = build_refactoring_examples(
        val_entries,
        args.model_name,
        args.max_source_length,
        args.max_target_length,
        prefix=not args.no_prefix,
    )

    config = CodeT5Config(
        model_name=args.model_name,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        prefix=not args.no_prefix,
    )
    model = CodeT5Refactor(config=config)
    model.train(train_examples, val_examples, args.output_dir, epochs=args.epochs, batch_size=args.batch_size)
    print(f"Saved refactoring model to {args.output_dir}")


if __name__ == "__main__":
    main()
