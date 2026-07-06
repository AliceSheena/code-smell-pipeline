import argparse
import ast
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from sklearn.model_selection import GroupKFold
except ImportError:
    GroupKFold = None


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            entries.append(json.loads(stripped))
    return entries


def normalize_code(code: str) -> str:
    try:
        tree = ast.parse(code)
        normalized = ast.dump(tree, include_attributes=False)
    except SyntaxError:
        normalized = "\n".join(line.strip() for line in code.splitlines() if line.strip())
    return normalized


def entry_fingerprint(entry: Dict[str, Any]) -> str:
    parts = [entry.get("smell_type", ""), entry.get("code_smell_code", ""), entry.get("refactoring_code", "")]
    normalized = "\n".join(parts)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return digest


def get_group_key(entry: Dict[str, Any]) -> str:
    for candidate in ["project", "project_name", "file", "file_path", "source_file"]:
        if entry.get(candidate):
            return str(entry[candidate])
    # fallback to smell_type + fingerprint prefix so duplicate-like entries stay together
    return f"group-{entry.get('smell_type','unknown')}-{entry_fingerprint(entry)[:8]}"


def verify_entry(entry: Dict[str, Any], index: int = -1) -> Tuple[bool, Optional[str]]:
    if not isinstance(entry, dict):
        return False, "entry is not a JSON object"

    required = ["smell_type", "smell_location", "code_smell_code", "refactoring_code"]
    for key in required:
        if key not in entry:
            return False, f"missing required key: {key}"

    loc = entry["smell_location"]
    if not isinstance(loc, dict):
        return False, "smell_location is not an object"

    for key in ["start_line", "end_line", "smelly_lines"]:
        if key not in loc:
            return False, f"smell_location missing {key}"

    if not isinstance(loc["smelly_lines"], list):
        return False, "smelly_lines must be a list"

    if not isinstance(entry["code_smell_code"], str) or not isinstance(entry["refactoring_code"], str):
        return False, "code_smell_code and refactoring_code must be strings"

    try:
        tree = ast.parse(entry["code_smell_code"])
    except SyntaxError as exc:
        return False, f"code_smell_code parse failure: {exc}"

    max_line = len(entry["code_smell_code"].splitlines())
    if not (1 <= loc["start_line"] <= loc["end_line"] <= max_line):
        return False, "smell_location start/end out of bounds"

    for line in loc["smelly_lines"]:
        if not isinstance(line, int) or line < 1 or line > max_line:
            return False, f"invalid smelly line number: {line}"

    return True, None


def split_dataset(entries: List[Dict[str, Any]], output_dir: Path, n_splits: int = 5, seed: int = 42) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = [get_group_key(e) for e in entries]

    if GroupKFold is None:
        raise ImportError("scikit-learn is required for group-aware splitting. Install scikit-learn.")

    unique_groups = set(groups)
    if len(unique_groups) < n_splits:
        print(f"Warning: not enough distinct groups for {n_splits}-fold group split ({len(unique_groups)} groups). Using random 70/15/15 split instead.")
        shuffled = entries.copy()
        random.shuffle(shuffled)
        train_end = int(0.7 * len(shuffled))
        val_end = train_end + int(0.15 * len(shuffled))
        train_entries = shuffled[:train_end]
        val_entries = shuffled[train_end:val_end]
        test_entries = shuffled[val_end:]
        with (output_dir / "train.jsonl").open("w", encoding="utf-8") as out:
            for entry in train_entries:
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
        with (output_dir / "val.jsonl").open("w", encoding="utf-8") as out:
            for entry in val_entries:
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
        with (output_dir / "test.jsonl").open("w", encoding="utf-8") as out:
            for entry in test_entries:
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return

    gkf = GroupKFold(n_splits=n_splits)
    for fold_index, (_, test_idx) in enumerate(gkf.split(entries, groups=groups, y=[e.get("smell_type", "") for e in entries])):
        fold_dir = output_dir / f"fold_{fold_index + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        test_entries = [entries[i] for i in test_idx]
        with (fold_dir / "test.jsonl").open("w", encoding="utf-8") as out:
            for entry in test_entries:
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # also write one train/val/test split from the first fold for quick reference
    train_candidates = [e for i, e in enumerate(entries) if i not in set(test_idx)]
    val_size = max(1, len(train_candidates) // 7)
    train_entries = train_candidates[:-val_size]
    val_entries = train_candidates[-val_size:]
    with (output_dir / "train.jsonl").open("w", encoding="utf-8") as out:
        for entry in train_entries:
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with (output_dir / "val.jsonl").open("w", encoding="utf-8") as out:
        for entry in val_entries:
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with (output_dir / "test.jsonl").open("w", encoding="utf-8") as out:
        for entry in test_entries:
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")


def prepare_dataset(input_path: str, output_dir: str, recover: bool = False) -> None:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    entries = load_jsonl(input_path)

    clean_entries = []
    dirty_reports = []
    fingerprints = {}

    for idx, entry in enumerate(entries):
        ok, reason = verify_entry(entry, idx)
        if not ok:
            dirty_reports.append((idx, reason, entry))
            continue

        fingerprint = entry_fingerprint(entry)
        if fingerprint in fingerprints:
            dirty_reports.append((idx, f"duplicate of entry {fingerprints[fingerprint]}", entry))
            continue

        fingerprints[fingerprint] = idx
        clean_entries.append(entry)

    report_path = output_dir / "prepare_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump({"clean_count": len(clean_entries), "dirty_count": len(dirty_reports), "dirty_reports": dirty_reports}, report_file, indent=2)

    if not clean_entries:
        raise RuntimeError("No valid dataset entries found after verification.")

    split_dataset(clean_entries, output_dir / "splits")
    print(f"Prepared {len(clean_entries)} clean entries and wrote splits to {output_dir / 'splits'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and split a code smell dataset.")
    parser.add_argument("--input", required=True, help="Path to the JSONL dataset file.")
    parser.add_argument("--output-dir", default="./data", help="Directory to save verified dataset and splits.")
    parser.add_argument("--recover", action="store_true", help="Attempt lightweight recovery of invalid examples.")
    args = parser.parse_args()

    prepare_dataset(args.input, args.output_dir, args.recover)


if __name__ == "__main__":
    main()
