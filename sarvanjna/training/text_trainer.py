"""
PyTorch Lightning trainer for Text-to-Text models.
"""

import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from typing import Optional, Dict, Any
import wandb

from sarvanjna.models.text.text_to_text import TextToTextModel
from sarvanjna.models.text.transformer import TransformerConfig


class TextToTextTrainer(pl.LightningModule):
    """
    PyTorch Lightning module for training Text-to-Text models.
    
    Handles:
    - Distributed training (DDP, FSDP)
    - Mixed precision
    - Gradient clipping
    - Learning rate scheduling
    - Checkpointing
    - Logging to WandB
    """
    
    def __init__(
        self,
        config: TransformerConfig,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 10000,
        max_steps: int = 1000000,
        gradient_clip: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = TextToTextModel(config)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.gradient_clip = gradient_clip
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
    
    def forward(self, **kwargs):
        """Forward pass."""
        return self.model(**kwargs)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Forward pass
        outputs = self.model(
            input_ids=batch["input_ids"],
            decoder_input_ids=batch.get("decoder_input_ids"),
            attention_mask=batch.get("attention_mask"),
            decoder_attention_mask=batch.get("decoder_attention_mask"),
            labels=batch.get("labels"),
        )
        
        loss = outputs["loss"]
        
        # Log metrics
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/learning_rate", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        
        # Track loss
        self.train_losses.append(loss.item())
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            decoder_input_ids=batch.get("decoder_input_ids"),
            attention_mask=batch.get("attention_mask"),
            decoder_attention_mask=batch.get("decoder_attention_mask"),
            labels=batch.get("labels"),
        )
        
        loss = outputs["loss"]
        
        # Log metrics
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        
        # Track loss
        self.val_losses.append(loss.item())
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight", "norm"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        
        # AdamW optimizer
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Learning rate scheduler with warmup
        def lr_lambda(current_step: int):
            if current_step < self.warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, self.warmup_steps))
            else:
                # Cosine decay
                progress = float(current_step - self.warmup_steps) / float(
                    max(1, self.max_steps - self.warmup_steps)
                )
                return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159))))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        if self.train_losses:
            avg_loss = sum(self.train_losses) / len(self.train_losses)
            self.log("train/epoch_loss", avg_loss)
            self.train_losses = []
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        if self.val_losses:
            avg_loss = sum(self.val_losses) / len(self.val_losses)
            self.log("val/epoch_loss", avg_loss)
            self.val_losses = []
    
    @torch.no_grad()
    def generate_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        **kwargs,
    ) -> torch.Tensor:
        """Generate text (inference)."""
        self.model.eval()
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            **kwargs,
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "total_params": self.model.get_num_params(),
            "trainable_params": self.model.get_num_trainable_params(),
            "config": self.model.config.__dict__,
        }
