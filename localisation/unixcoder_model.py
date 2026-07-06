import bisect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_scheduler

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None

logger = logging.getLogger(__name__)


@dataclass
class UniXcoderConfig:
    model_name: str = "microsoft/unixcoder-base"
    max_length: int = 256
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    num_labels: int = 2
    alpha: float = 1.0
    beta: float = 1.0
    learning_rate: float = 3e-5
    weight_decay: float = 0.01


class DualHeadUnixcoderModel(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int, num_labels: int = 2, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.token_classifier = nn.Linear(hidden_size, 2)
        self.alpha = alpha
        self.beta = beta

    def forward(self, input_ids, attention_mask=None, labels=None, token_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        cls_state = hidden_states[:, 0, :]
        classification_logits = self.classifier(cls_state)
        token_logits = self.token_classifier(hidden_states)

        loss = None
        if labels is not None and token_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            cls_loss = loss_fct(classification_logits.view(-1, classification_logits.size(-1)), labels.view(-1))
            token_loss = loss_fct(token_logits.view(-1, 2), token_labels.view(-1))
            loss = self.alpha * token_loss + self.beta * cls_loss

        return {
            "loss": loss,
            "logits": classification_logits,
            "token_logits": token_logits,
        }


class LocalisationDataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        return {
            "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(example["attention_mask"], dtype=torch.long),
            "classification_label": torch.tensor(example["classification_label"], dtype=torch.long),
            "token_labels": torch.tensor(example["token_labels"], dtype=torch.long),
        }


class UniXcoderLocaliser:
    def __init__(self, config: UniXcoderConfig = UniXcoderConfig()):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        encoder = AutoModel.from_pretrained(self.config.model_name)
        hidden_size = encoder.config.hidden_size
        base_model = DualHeadUnixcoderModel(
            encoder,
            hidden_size,
            num_labels=self.config.num_labels,
            alpha=self.config.alpha,
            beta=self.config.beta,
        )
        if get_peft_model is not None:
            peft_config = LoraConfig(
                task_type="SEQ_CLASSIFICATION",
                inference_mode=False,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["query", "value"],
            )
            self.model = get_peft_model(base_model, peft_config)
        else:
            self.model = base_model

    @staticmethod
    def _token_to_line(offsets: List[tuple], source: str) -> List[int]:
        line_boundaries = [0]
        for line in source.splitlines(keepends=True):
            line_boundaries.append(line_boundaries[-1] + len(line))

        line_ids: List[int] = []
        for offset in offsets:
            if offset is None or offset[0] == offset[1]:
                line_ids.append(0)
                continue
            pos = offset[0]
            line = bisect.bisect_right(line_boundaries, pos)
            line_ids.append(max(1, min(line, len(line_boundaries) - 1)))
        return line_ids

    def _decode_lines_from_tokens(self, token_preds: List[int], offsets: List[tuple], source: str) -> List[int]:
        line_ids = self._token_to_line(offsets, source)
        smelly_lines = sorted({line_ids[idx] for idx, label in enumerate(token_preds) if label == 1 and line_ids[idx] > 0})
        return smelly_lines

    def _collate(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        inputs = [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in batch]
        batch_inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        token_labels = pad_sequence([item["token_labels"] for item in batch], batch_first=True, padding_value=-100)
        batch_inputs["token_labels"] = token_labels
        batch_inputs["labels"] = torch.stack([item["classification_label"] for item in batch])
        return batch_inputs

    def train(self, train_examples: List[Dict[str, Any]], val_examples: List[Dict[str, Any]], output_dir: str, epochs: int = 3, batch_size: int = 2):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        train_loader = DataLoader(LocalisationDataset(train_examples), batch_size=batch_size, shuffle=True, collate_fn=self._collate)
        val_loader = DataLoader(LocalisationDataset(val_examples), batch_size=batch_size, shuffle=False, collate_fn=self._collate)

        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        total_steps = len(train_loader) * epochs
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    token_labels=batch["token_labels"],
                )
                loss = outputs["loss"]
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader) if train_loader else 0.0
            val_metrics = self.evaluate(val_examples)
            logger.info(f"Epoch {epoch} | train_loss={avg_loss:.4f} | val_metrics={val_metrics}")
            self.save(output_dir)

    def evaluate(self, examples: List[Dict[str, Any]]) -> Dict[str, float]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0
        correct_tokens = 0
        with torch.no_grad():
            for example in examples:
                input_ids = torch.tensor(example["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
                attention_mask = torch.tensor(example["attention_mask"], dtype=torch.long).unsqueeze(0).to(device)
                labels = torch.tensor([example["classification_label"]], dtype=torch.long).to(device)
                token_labels = torch.tensor(example["token_labels"], dtype=torch.long).unsqueeze(0).to(device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, token_labels=token_labels)
                loss = outputs["loss"]
                total_loss += loss.item()
                preds = torch.argmax(outputs["token_logits"], dim=-1).squeeze(0).cpu().tolist()
                token_ref = example["token_labels"]
                token_ref_filtered = [t for t in token_ref if t != -100]
                preds_filtered = [p for p, t in zip(preds, token_ref) if t != -100]
                total_tokens += len(preds_filtered)
                correct_tokens += sum(int(p == t) for p, t in zip(preds_filtered, token_ref_filtered))

        accuracy = correct_tokens / total_tokens if total_tokens else 0.0
        return {"avg_loss": total_loss / len(examples) if examples else 0.0, "token_accuracy": accuracy}

    def save(self, output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(output_path)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_path)

    def predict(self, source_code: str) -> Dict[str, Any]:
        encoding = self.tokenizer(
            source_code,
            truncation=True,
            max_length=self.config.max_length,
            return_offsets_mapping=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        inputs = {k: v.to(device) for k, v in encoding.items() if k in ["input_ids", "attention_mask"]}
        outputs = self.model(**inputs)
        token_predictions = torch.argmax(outputs["token_logits"], dim=-1).squeeze(0).cpu().tolist()
        offsets = encoding["offset_mapping"].squeeze(0).tolist()
        smelly_lines = self._decode_lines_from_tokens(token_predictions, offsets, source_code)
        return {
            "smell_type": "unknown",
            "smelly_lines": smelly_lines,
            "confidence": 0.5,
        }
