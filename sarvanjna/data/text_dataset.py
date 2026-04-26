"""
Text dataset for language modeling and text-to-text tasks.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import torch
from torch.utils.data import Dataset
import json


class TextDataset(Dataset):
    """
    Dataset for text processing tasks.
    
    Supports:
    - Raw text corpora
    - Instruction-response pairs
    - Summarization datasets
    - QA datasets
    """
    
    def __init__(
        self,
        data_path: Path,
        tokenizer: Optional[Any] = None,
        max_length: int = 512,
        task_type: str = "lm",  # lm, instruction, summarization, qa
        split: str = "train",
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        self.split = split
        
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load text data from disk."""
        if self.data_path.suffix == ".json":
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif self.data_path.suffix == ".jsonl":
            data = []
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))
        elif self.data_path.suffix == ".txt":
            with open(self.data_path, "r", encoding="utf-8") as f:
                text = f.read()
            # Split into chunks for language modeling
            data = [{"text": text}]
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        item = self.data[idx]
        
        if self.task_type == "lm":
            # Language modeling: just the text
            text = item.get("text", "")
            if self.tokenizer:
                encoded = self.tokenizer.encode(text, max_length=self.max_length)
                return {
                    "input_ids": torch.tensor(encoded.ids),
                    "attention_mask": torch.tensor(encoded.attention_mask),
                }
            return {"text": text}
        
        elif self.task_type == "instruction":
            # Instruction following: input + output
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")
            
            # Format as: instruction [SEP] input -> output
            full_input = f"{instruction} {input_text}".strip()
            
            if self.tokenizer:
                input_encoded = self.tokenizer.encode(full_input, max_length=self.max_length)
                output_encoded = self.tokenizer.encode(output, max_length=self.max_length)
                
                return {
                    "input_ids": torch.tensor(input_encoded.ids),
                    "attention_mask": torch.tensor(input_encoded.attention_mask),
                    "labels": torch.tensor(output_encoded.ids),
                }
            
            return {
                "input": full_input,
                "output": output,
            }
        
        elif self.task_type == "summarization":
            # Summarization: document -> summary
            document = item.get("document", "")
            summary = item.get("summary", "")
            
            if self.tokenizer:
                doc_encoded = self.tokenizer.encode(document, max_length=self.max_length)
                sum_encoded = self.tokenizer.encode(summary, max_length=self.max_length)
                
                return {
                    "input_ids": torch.tensor(doc_encoded.ids),
                    "attention_mask": torch.tensor(doc_encoded.attention_mask),
                    "labels": torch.tensor(sum_encoded.ids),
                }
            
            return {
                "document": document,
                "summary": summary,
            }
        
        elif self.task_type == "qa":
            # Question answering: question + context -> answer
            question = item.get("question", "")
            context = item.get("context", "")
            answer = item.get("answer", "")
            
            full_input = f"Question: {question} Context: {context}"
            
            if self.tokenizer:
                input_encoded = self.tokenizer.encode(full_input, max_length=self.max_length)
                answer_encoded = self.tokenizer.encode(answer, max_length=self.max_length)
                
                return {
                    "input_ids": torch.tensor(input_encoded.ids),
                    "attention_mask": torch.tensor(input_encoded.attention_mask),
                    "labels": torch.tensor(answer_encoded.ids),
                }
            
            return {
                "input": full_input,
                "answer": answer,
            }
        
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for batching."""
        if not batch:
            return {}
        
        # Handle different batch types
        if "input_ids" in batch[0]:
            # Tokenized batch
            max_input_len = max(item["input_ids"].size(0) for item in batch)
            max_label_len = max(item.get("labels", torch.tensor([])).size(0) for item in batch)
            
            input_ids = []
            attention_masks = []
            labels = []
            
            for item in batch:
                # Pad input
                input_len = item["input_ids"].size(0)
                pad_len = max_input_len - input_len
                input_ids.append(
                    torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
                )
                attention_masks.append(
                    torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
                )
                
                # Pad labels if present
                if "labels" in item:
                    label_len = item["labels"].size(0)
                    label_pad_len = max_label_len - label_len
                    labels.append(
                        torch.cat([item["labels"], torch.full((label_pad_len,), -100, dtype=torch.long)])
                    )
            
            result = {
                "input_ids": torch.stack(input_ids),
                "attention_mask": torch.stack(attention_masks),
            }
            
            if labels:
                result["labels"] = torch.stack(labels)
            
            return result
        
        else:
            # Text batch (no tokenization)
            return batch
