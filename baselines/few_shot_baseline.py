from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass
class FewShotBaselineConfig:
    model_name: str = "Salesforce/codet5-base"
    max_source_length: int = 256
    max_target_length: int = 256
    num_beams: int = 3
    few_shot_examples: Optional[List[Dict[str, str]]] = None


class FewShotBaseline:
    def __init__(self, config: FewShotBaselineConfig = FewShotBaselineConfig()):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)

    def _build_prompt(self, smell_type: str, source_code: str) -> str:
        prompt_lines = [
            "Detect the smelly lines and produce a fix.",
            "",
        ]
        if self.config.few_shot_examples:
            for example in self.config.few_shot_examples:
                prompt_lines.append(f"Smell type: {example['smell_type']}")
                prompt_lines.append(example["code_smell_code"])
                prompt_lines.append("---")
        prompt_lines.append(f"Smell type: {smell_type}")
        prompt_lines.append(source_code)
        return "\n".join(prompt_lines)

    def predict_refactor(self, smell_type: str, source_code: str) -> str:
        prompt = self._build_prompt(smell_type, source_code)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_source_length)
        outputs = self.model.generate(
            **inputs,
            max_length=self.config.max_target_length,
            num_beams=self.config.num_beams,
            early_stopping=True,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def predict_localisation(self, smell_type: str, source_code: str) -> Dict[str, Any]:
        return {
            "smell_type": smell_type,
            "smelly_lines": [],
            "confidence": 0.0,
            "notes": "Non-fine-tuned weak baseline; implement task-specific prompt for localisation.",
        }
