import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_scheduler

logger = logging.getLogger(__name__)


@dataclass
class CodeT5Config:
    model_name: str = "Salesforce/codet5-base"
    max_source_length: int = 256
    max_target_length: int = 256
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    beam_size: int = 5
    prefix: bool = True
    learning_rate: float = 3e-5
    weight_decay: float = 0.01


class RefactorDataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        return {
            "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(example["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(example["labels"], dtype=torch.long),
        }


class CodeT5Refactor:
    def __init__(self, config: CodeT5Config = CodeT5Config()):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)

    def _build_input(self, smell_type: str, source_code: str) -> str:
        if self.config.prefix:
            return f"[SMELL_TYPE] {smell_type}\n{source_code}"
        return source_code

    def _collate(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        inputs = [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in batch]
        batch_inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
        batch_inputs["labels"] = labels
        return batch_inputs

    def generate(self, smell_type: str, source_code: str) -> str:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        prompt = self._build_input(smell_type, source_code)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_source_length).to(device)
        outputs = self.model.generate(
            **inputs,
            max_length=self.config.max_target_length,
            num_beams=self.config.beam_size,
            early_stopping=True,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def train(self, train_examples: List[Dict[str, Any]], val_examples: List[Dict[str, Any]], output_dir: str, epochs: int = 3, batch_size: int = 2):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        train_loader = DataLoader(RefactorDataset(train_examples), batch_size=batch_size, shuffle=True, collate_fn=self._collate)
        val_loader = DataLoader(RefactorDataset(val_examples), batch_size=batch_size, shuffle=False, collate_fn=self._collate)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        total_steps = len(train_loader) * epochs
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader) if train_loader else 0.0
            val_loss = self.evaluate(val_examples)
            logger.info(f"Epoch {epoch} | train_loss={avg_loss:.4f} | val_loss={val_loss:.4f}")
            self.save(output_dir)

    def evaluate(self, examples: List[Dict[str, Any]]) -> float:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        total_loss = 0.0
        with torch.no_grad():
            for example in examples:
                input_ids = torch.tensor(example["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
                attention_mask = torch.tensor(example["attention_mask"], dtype=torch.long).unsqueeze(0).to(device)
                labels = torch.tensor(example["labels"], dtype=torch.long).unsqueeze(0).to(device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()

        return total_loss / len(examples) if examples else 0.0

    def save(self, output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(output_path)
        self.model.save_pretrained(output_path)
