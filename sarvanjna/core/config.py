"""
Core configuration management for the Sarvanjna platform.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import os


@dataclass
class DataConfig:
    """Data layer configuration."""
    raw_data_path: Path = Path("data/raw")
    processed_data_path: Path = Path("data/processed")
    cache_dir: Path = Path("data/cache")
    max_workers: int = 8
    batch_size: int = 32
    

@dataclass
class TextConfig:
    """Text processing configuration."""
    vocab_size: int = 32000
    max_seq_length: int = 512
    sentencepiece_model_type: str = "unigram"  # or "bpe"
    normalization: str = "nmt_nfkc"
    
    
@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Text-to-Text
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    
    # Image
    image_size: int = 512
    latent_dim: int = 4
    vae_scale_factor: int = 8
    
    # Video
    video_frames: int = 16
    frame_rate: int = 24
    
    # Audio
    audio_sample_rate: int = 32000
    audio_channels: int = 1


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10000
    max_steps: int = 1000000
    gradient_clip: float = 1.0
    
    # Distributed
    strategy: str = "ddp"  # ddp, fsdp, deepspeed
    precision: str = "bf16-mixed"  # 32, 16-mixed, bf16-mixed
    num_nodes: int = 1
    devices: int = -1  # -1 means all available
    
    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    save_every_n_steps: int = 5000
    keep_last_n_checkpoints: int = 3
    
    # Logging
    log_every_n_steps: int = 100
    val_check_interval: int = 1000
    use_wandb: bool = True
    wandb_project: str = "sarvanjna"


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    text: TextConfig = field(default_factory=TextConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    seed: int = 42
    debug: bool = False
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        config_dict = self._to_dict()
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "data": self.data.__dict__,
            "text": self.text.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "seed": self.seed,
            "debug": self.debug,
        }
    
    def setup_directories(self):
        """Create necessary directories."""
        dirs = [
            self.data.raw_data_path,
            self.data.processed_data_path,
            self.data.cache_dir,
            self.training.checkpoint_dir,
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
