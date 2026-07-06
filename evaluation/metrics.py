import ast
from collections import Counter
from typing import Dict, List, Tuple


def line_iou(pred_ranges: List[Tuple[int, int]], gold_ranges: List[Tuple[int, int]]) -> float:
    pred_lines = set()
    gold_lines = set()
    for start, end in pred_ranges:
        pred_lines.update(range(start, end + 1))
    for start, end in gold_ranges:
        gold_lines.update(range(start, end + 1))

    if not gold_lines and not pred_lines:
        return 1.0
    if not gold_lines or not pred_lines:
        return 0.0

    intersection = len(pred_lines & gold_lines)
    union = len(pred_lines | gold_lines)
    return intersection / union if union else 0.0


def exact_match(pred: str, gold: str) -> bool:
    return pred.strip() == gold.strip()


def token_f1(pred_tokens: List[int], gold_tokens: List[int]) -> float:
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_count = Counter(pred_tokens)
    gold_count = Counter(gold_tokens)
    common = sum(min(pred_count[t], gold_count[t]) for t in pred_count)
    precision = common / sum(pred_count.values())
    recall = common / sum(gold_count.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def parse_line_ranges(lines: List[int]) -> List[Tuple[int, int]]:
    if not lines:
        return []
    sorted_lines = sorted(set(lines))
    ranges = []
    start = sorted_lines[0]
    end = start
    for line in sorted_lines[1:]:
        if line == end + 1:
            end = line
        else:
            ranges.append((start, end))
            start = line
            end = line
    ranges.append((start, end))
    return ranges


def codebleu_score(pred: str, gold: str, lang: str = "python") -> float:
    try:
        from codebleu import calc_code_bleu
    except ImportError:
        return 0.0

    results = calc_code_bleu([pred], [gold], lang)
    return float(results.get("CodeBLEU", 0.0))


def ast_is_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
