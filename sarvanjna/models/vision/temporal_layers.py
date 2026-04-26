"""
Temporal layers for video diffusion models.

Based on "Stable Video Diffusion" (Blattmann et al., 2023).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class TemporalAttention(nn.Module):
    """
    Temporal self-attention layer for video.
    
    Processes sequences across time dimension while keeping spatial dimensions.
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj_out = nn.Linear(channels, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with temporal attention.
        
        Args:
            x: (batch, channels, frames, height, width)
        
        Returns:
            out: (batch, channels, frames, height, width)
        """
        B, C, F, H, W = x.shape
        residual = x
        
        # Reshape for temporal attention
        x = x.permute(0, 2, 3, 4, 1)  # (B, F, H, W, C)
        x = x.reshape(B * H * W, F, C)  # (B*H*W, F, C)
        
        # Apply norm
        x_norm = self.norm(residual)
        x_norm = x_norm.permute(0, 2, 3, 4, 1).reshape(B * H * W, F, C)
        
        # QKV projection
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(B * H * W, F, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*H*W, num_heads, F, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        out = attn @ v
        
        # Reshape back
        out = out.transpose(1, 2).reshape(B * H * W, F, C)
        out = self.proj_out(out)
        
        # Reshape to original shape
        out = out.reshape(B, H, W, F, C).permute(0, 4, 3, 1, 2)  # (B, C, F, H, W)
        
        return out + residual


class TemporalConv3D(nn.Module):
    """
    3D convolution for temporal processing.
    
    Applies convolution across time and space.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.GroupNorm(32, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, channels, frames, height, width)
        
        Returns:
            out: (batch, out_channels, frames, height, width)
        """
        x = self.conv(x)
        x = self.norm(x)
        x = F.silu(x)
        return x


class TemporalResBlock(nn.Module):
    """
    Residual block with temporal processing.
    
    Combines spatial and temporal convolutions.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Spatial processing (2D conv on each frame)
        self.spatial_norm1 = nn.GroupNorm(32, in_channels)
        self.spatial_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding
        self.time_emb_proj = nn.Linear(time_embed_dim, out_channels)
        
        # Temporal processing
        self.temporal_norm = nn.GroupNorm(32, out_channels)
        self.temporal_conv = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
        )
        
        # Second spatial conv
        self.spatial_norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.spatial_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, in_channels, frames, height, width)
            time_emb: (batch, time_embed_dim)
        
        Returns:
            out: (batch, out_channels, frames, height, width)
        """
        B, C, F, H, W = x.shape
        residual = x
        
        # Spatial conv 1 (process each frame)
        h = self.spatial_norm1(x)
        h = F.silu(h)
        h = h.reshape(B * F, C, H, W)
        h = self.spatial_conv1(h)
        h = h.reshape(B, -1, F, H, W)
        
        # Add time embedding
        time_emb = self.time_emb_proj(F.silu(time_emb))
        h = h + time_emb[:, :, None, None, None]
        
        # Temporal conv
        h = self.temporal_norm(h)
        h = F.silu(h)
        h = self.temporal_conv(h)
        
        # Spatial conv 2
        h = self.spatial_norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = h.reshape(B * F, -1, H, W)
        h = self.spatial_conv2(h)
        h = h.reshape(B, -1, F, H, W)
        
        # Shortcut
        residual = residual.reshape(B * F, C, H, W)
        residual = self.shortcut(residual)
        residual = residual.reshape(B, -1, F, H, W)
        
        return h + residual


class PositionalEmbedding3D(nn.Module):
    """3D positional embedding for video."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
    
    def forward(self, shape: tuple) -> torch.Tensor:
        """
        Generate 3D positional embeddings.
        
        Args:
            shape: (frames, height, width)
        
        Returns:
            embeddings: (channels, frames, height, width)
        """
        F, H, W = shape
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Generate positional encodings
        channels = self.channels // 3
        
        # Temporal
        t_pos = torch.arange(F, device=device).float()
        t_emb = self._get_embeddings(t_pos, channels)  # (F, channels)
        t_emb = t_emb[:, :, None, None].expand(F, channels, H, W)
        
        # Height
        h_pos = torch.arange(H, device=device).float()
        h_emb = self._get_embeddings(h_pos, channels)  # (H, channels)
        h_emb = h_emb[None, :, :, None].expand(F, channels, H, W)
        h_emb = h_emb.transpose(0, 2).transpose(2, 3)  # Rearrange
        
        # Width
        w_pos = torch.arange(W, device=device).float()
        w_emb = self._get_embeddings(w_pos, channels)  # (W, channels)
        w_emb = w_emb[None, None, :, :].expand(F, H, channels, W)
        w_emb = w_emb.transpose(0, 2).transpose(2, 3)  # Rearrange
        
        # Concatenate
        embeddings = torch.cat([t_emb, h_emb, w_emb], dim=1)  # (F, channels*3, H, W)
        embeddings = embeddings.transpose(0, 1)  # (channels*3, F, H, W)
        
        return embeddings
    
    def _get_embeddings(self, positions: torch.Tensor, dim: int) -> torch.Tensor:
        """Generate sinusoidal embeddings."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=positions.device) * -emb)
        emb = positions[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
