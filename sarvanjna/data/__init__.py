"""
Data layer for the multimodal AI platform.

Handles ingestion, versioning, and lineage tracking for:
- Text corpora
- Image-text pairs
- Video-text pairs
- Audio/music-text pairs
"""

from sarvanjna.data.text_dataset import TextDataset
from sarvanjna.data.image_text_dataset import ImageTextDataset
from sarvanjna.data.data_manager import DataManager

__all__ = ["TextDataset", "ImageTextDataset", "DataManager"]
