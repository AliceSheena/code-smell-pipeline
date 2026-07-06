import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            entries.append(json.loads(stripped))
    return entries


def save_jsonl(path: Path, entries: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _line_boundaries(code: str) -> List[int]:
    boundaries = [0]
    for line in code.splitlines(keepends=True):
        boundaries.append(boundaries[-1] + len(line))
    return boundaries


def _line_intervals(code: str, smelly_lines: List[int]) -> List[Tuple[int, int]]:
    boundaries = _line_boundaries(code)
    intervals = []
    for line in sorted(set(smelly_lines)):
        if 1 <= line < len(boundaries):
            start = boundaries[line - 1]
            end = boundaries[line]
            intervals.append((start, end))
    return intervals


def build_token_labels(code: str, smelly_lines: List[int], tokenizer: AutoTokenizer, max_length: int) -> Dict[str, Any]:
    encoded = tokenizer(
        code,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_attention_mask=True,
    )
    offsets = encoded.pop("offset_mapping")
    line_intervals = _line_intervals(code, smelly_lines)
    labels = []
    for offset in offsets:
        if offset is None or offset[0] == offset[1]:
            labels.append(-100)
            continue
        label = 0
        for start, end in line_intervals:
            if not (offset[1] <= start or offset[0] >= end):
                label = 1
                break
        labels.append(label)
    encoded["token_labels"] = labels
    return encoded


def build_localisation_examples(entries: List[Dict[str, Any]], model_name: str, max_length: int) -> List[Dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    examples: List[Dict[str, Any]] = []
    for entry in entries:
        code = entry["code_smell_code"]
        smelly_lines = entry["smell_location"].get("smelly_lines", [])
        encoding = build_token_labels(code, smelly_lines, tokenizer, max_length)
        encoding["classification_label"] = 1
        encoding["metadata"] = {
            "smell_type": entry["smell_type"],
            "code_smell_code": code,
            "smelly_lines": smelly_lines,
        }
        examples.append(encoding)
    return examples


def build_refactoring_examples(entries: List[Dict[str, Any]], model_name: str, source_max_length: int, target_max_length: int, prefix: bool = True) -> List[Dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    examples: List[Dict[str, Any]] = []
    for entry in entries:
        source = entry["code_smell_code"]
        smell_type = entry["smell_type"]
        if prefix:
            source = f"[SMELL_TYPE] {smell_type}\n{source}"
        target = entry["refactoring_code"]
        source_encoding = tokenizer(source, truncation=True, max_length=source_max_length, return_attention_mask=True)
        target_encoding = tokenizer(target, truncation=True, max_length=target_max_length, return_attention_mask=False)
        labels = target_encoding["input_ids"].copy()
        labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]
        examples.append({
            "input_ids": source_encoding["input_ids"],
            "attention_mask": source_encoding["attention_mask"],
            "labels": labels,
            "metadata": {
                "smell_type": smell_type,
                "code_smell_code": entry["code_smell_code"],
            },
        })
    return examples
