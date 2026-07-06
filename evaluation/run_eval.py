import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import codebleu_score, exact_match, line_iou, parse_line_ranges


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation on predictions against references.")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSONL file.")
    parser.add_argument("--references", required=True, help="Path to reference JSONL file.")
    args = parser.parse_args()

    preds = load_jsonl(Path(args.predictions))
    refs = load_jsonl(Path(args.references))

    if len(preds) != len(refs):
        raise ValueError("Predictions and references must have equal length.")

    total = len(preds)
    iou_sum = 0.0
    codebleu_sum = 0.0
    exact_count = 0

    for pred, ref in zip(preds, refs):
        pred_ranges = parse_line_ranges(pred.get("smelly_lines", []))
        gold_ranges = parse_line_ranges(ref["smell_location"].get("smelly_lines", []))
        iou_sum += line_iou(pred_ranges, gold_ranges)
        codebleu_sum += codebleu_score(pred.get("refactoring_code", ""), ref.get("refactoring_code", ""))
        exact_count += int(exact_match(pred.get("refactoring_code", ""), ref.get("refactoring_code", "")))

    print(f"examples: {total}")
    print(f"average line IoU: {iou_sum / total:.4f}")
    print(f"average CodeBLEU: {codebleu_sum / total:.4f}")
    print(f"exact match: {exact_count}/{total} = {exact_count/total:.4f}")


if __name__ == "__main__":
    main()
