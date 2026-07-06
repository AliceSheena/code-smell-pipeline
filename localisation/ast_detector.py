import ast
from typing import Dict, List, Optional, Tuple

INPLACE_METHOD_SUFFIXES = {
    "add_",
    "sub_",
    "mul_",
    "div_",
    "pow_",
    "zero_",
    "fill_",
    "resize_",
    "copy_",
    "relu_",
    "sigmoid_",
}


def _extract_line(node: ast.AST) -> int:
    return getattr(node, "lineno", -1)


def _is_inplace_method_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute):
        name = func.attr
        if name in INPLACE_METHOD_SUFFIXES:
            return True
    return False


def _has_inplace_keyword(node: ast.Call) -> bool:
    for keyword in node.keywords:
        if keyword.arg == "inplace" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
            return True
    return False


def detect_inplace_api_misuse(source_code: str) -> Dict[str, object]:
    lines: List[int] = []
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return {
            "smell_type": "In-Place APIs Misused",
            "smelly_lines": [],
            "smelly_ranges": [],
            "confidence": 0.0,
        }

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if _is_inplace_method_call(node) or _has_inplace_keyword(node):
                line = _extract_line(node)
                if line > 0:
                    lines.append(line)

    lines = sorted(set(lines))
    ranges = []
    if lines:
        start = lines[0]
        end = lines[0]
        for line in lines[1:]:
            if line == end + 1:
                end = line
            else:
                ranges.append((start, end))
                start = line
                end = line
        ranges.append((start, end))

    return {
        "smell_type": "In-Place APIs Misused",
        "smelly_lines": lines,
        "smelly_ranges": ranges,
        "confidence": 0.9 if lines else 0.0,
    }


class ASTInPlaceAPIMisuseDetector:
    def __init__(self, rules: Optional[Dict[str, object]] = None):
        self.rules = rules or {}

    def predict(self, source_code: str) -> Dict[str, object]:
        return detect_inplace_api_misuse(source_code)
