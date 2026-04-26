"""
Vision Transformer (ViT) implementation.

Based on "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
(Dosovitskiy et al., 2021) - https://arxiv.org/abs/2010.11929
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class ViTConfig:
    """Vision Transformer configuration."""
    
    # Image settings
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    
    # Model architecture
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    
    # Output
    num_classes: Optional[int] = None  # For classification, None for embeddings
    pool_type: str = "cls"  # "cls" or "mean"
    
    def __post_init__(self):
        assert self.image_size % self.patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (self.image_size // self.patch_size) ** 2


class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings."""
    
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        
        # Convolutional projection of patches
        self.projection = nn.Conv2d(
            config.in_channels,
            config.d_model,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, height, width)
        
        Returns:
            patches: (batch_size, num_patches, d_model)
        """
        # x: (B, C, H, W) -> (B, d_model, H//P, W//P)
        x = self.projection(x)
        
        # Flatten patches: (B, d_model, H//P, W//P) -> (B, d_model, num_patches)
        x = x.flatten(2)
        
        # Transpose: (B, d_model, num_patches) -> (B, num_patches, d_model)
        x = x.transpose(1, 2)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for ViT."""
    
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        
        self.qkv = nn.Linear(config.d_model, config.d_model * 3)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, N, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = self.d_head ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, n_heads, N, N)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block for ViT."""
    
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Self-attention with residual
        x = x + self.attn(self.norm1(x))
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for image encoding.
    
    Converts images into sequence of patch embeddings and processes with
    Transformer encoder blocks.
    """
    
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(config)
        
        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.d_model)
        )
        self.pos_drop = nn.Dropout(config.dropout)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        self.norm = nn.LayerNorm(config.d_model)
        
        # Optional classification head
        if config.num_classes is not None:
            self.head = nn.Linear(config.d_model, config.num_classes)
        else:
            self.head = nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Initialize positional embeddings with truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize other layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through Vision Transformer.
        
        Args:
            x: (batch_size, channels, height, width) input images
            return_features: If True, return sequence features instead of pooled
        
        Returns:
            If return_features:
                features: (batch_size, num_patches+1, d_model)
            Else:
                output: (batch_size, d_model) or (batch_size, num_classes)
        """
        B = x.shape[0]
        
        # Convert image to patch embeddings
        x = self.patch_embed(x)  # (B, num_patches, d_model)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, d_model)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Return features if requested
        if return_features:
            return x
        
        # Pool features
        if self.config.pool_type == "cls":
            x = x[:, 0]  # Use CLS token
        elif self.config.pool_type == "mean":
            x = x[:, 1:].mean(dim=1)  # Mean pool over patches
        else:
            raise ValueError(f"Unknown pool_type: {self.config.pool_type}")
        
        # Apply classification head if present
        x = self.head(x)
        
        return x
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


# Predefined ViT configurations
def vit_tiny(image_size: int = 224, num_classes: Optional[int] = None) -> VisionTransformer:
    """ViT-Tiny: ~5M parameters."""
    config = ViTConfig(
        image_size=image_size,
        patch_size=16,
        d_model=192,
        n_heads=3,
        n_layers=12,
        d_ff=768,
        num_classes=num_classes,
    )
    return VisionTransformer(config)


def vit_small(image_size: int = 224, num_classes: Optional[int] = None) -> VisionTransformer:
    """ViT-Small: ~22M parameters."""
    config = ViTConfig(
        image_size=image_size,
        patch_size=16,
        d_model=384,
        n_heads=6,
        n_layers=12,
        d_ff=1536,
        num_classes=num_classes,
    )
    return VisionTransformer(config)


def vit_base(image_size: int = 224, num_classes: Optional[int] = None) -> VisionTransformer:
    """ViT-Base: ~86M parameters."""
    config = ViTConfig(
        image_size=image_size,
        patch_size=16,
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072,
        num_classes=num_classes,
    )
    return VisionTransformer(config)


def vit_large(image_size: int = 224, num_classes: Optional[int] = None) -> VisionTransformer:
    """ViT-Large: ~304M parameters."""
    config = ViTConfig(
        image_size=image_size,
        patch_size=16,
        d_model=1024,
        n_heads=16,
        n_layers=24,
        d_ff=4096,
        num_classes=num_classes,
    )
    return VisionTransformer(config)
