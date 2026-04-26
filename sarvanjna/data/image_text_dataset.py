"""
Image-text paired dataset for vision-language tasks.
"""

from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import json


class ImageTextDataset(Dataset):
    """
    Dataset for image-text paired data.
    
    Supports:
    - Image captioning
    - Text-to-image generation (prompts)
    - Vision-language alignment (CLIP-style)
    """
    
    def __init__(
        self,
        data_path: Path,
        image_transform: Optional[Callable] = None,
        text_tokenizer: Optional[Any] = None,
        max_text_length: int = 77,
        return_raw: bool = False,
    ):
        self.data_path = Path(data_path)
        self.image_transform = image_transform
        self.text_tokenizer = text_tokenizer
        self.max_text_length = max_text_length
        self.return_raw = return_raw
        
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Load image-text pairs from disk.
        
        Expected format (JSON/JSONL):
        {
            "image_path": "path/to/image.jpg",
            "caption": "A description of the image",
            "metadata": {...}  # optional
        }
        """
        if self.data_path.suffix == ".json":
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif self.data_path.suffix == ".jsonl":
            data = []
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        # Validate image paths
        validated_data = []
        base_path = self.data_path.parent
        
        for item in data:
            image_path = base_path / item["image_path"]
            if image_path.exists():
                item["full_image_path"] = str(image_path)
                validated_data.append(item)
        
        print(f"Loaded {len(validated_data)}/{len(data)} valid image-text pairs")
        return validated_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single image-text pair."""
        item = self.data[idx]
        
        # Load image
        image_path = item["full_image_path"]
        image = Image.open(image_path).convert("RGB")
        
        # Get caption
        caption = item.get("caption", "")
        
        if self.return_raw:
            return {
                "image": image,
                "caption": caption,
                "metadata": item.get("metadata", {}),
            }
        
        # Apply transforms
        result = {}
        
        if self.image_transform:
            image = self.image_transform(image)
        result["image"] = image
        
        if self.text_tokenizer:
            # Tokenize caption
            encoded = self.text_tokenizer.encode(
                caption,
                max_length=self.max_text_length,
                padding="max_length",
                truncation=True,
            )
            result["input_ids"] = torch.tensor(encoded.ids)
            result["attention_mask"] = torch.tensor(encoded.attention_mask)
        else:
            result["caption"] = caption
        
        result["metadata"] = item.get("metadata", {})
        
        return result
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for batching."""
        if not batch:
            return {}
        
        result = {}
        
        # Stack images
        if "image" in batch[0] and torch.is_tensor(batch[0]["image"]):
            result["image"] = torch.stack([item["image"] for item in batch])
        
        # Stack text encodings
        if "input_ids" in batch[0]:
            result["input_ids"] = torch.stack([item["input_ids"] for item in batch])
            result["attention_mask"] = torch.stack([item["attention_mask"] for item in batch])
        elif "caption" in batch[0]:
            result["caption"] = [item["caption"] for item in batch]
        
        # Collect metadata
        result["metadata"] = [item.get("metadata", {}) for item in batch]
        
        return result
