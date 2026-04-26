"""
U-Net architecture for diffusion models.

Based on "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
and "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List
import math


@dataclass
class UNetConfig:
    """U-Net configuration for diffusion."""
    
    # Input/output
    in_channels: int = 4  # Latent channels
    out_channels: int = 4
    
    # Architecture
    model_channels: int = 320
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (4, 2, 1)  # At which stages to use attention
    dropout: float = 0.0
    
    # Conditioning
    context_dim: int = 768  # Text embedding dimension
    num_heads: int = 8
    
    # Time embedding
    time_embed_dim: int = 1280


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            timesteps: (batch_size,) tensor of timesteps
        
        Returns:
            embeddings: (batch_size, dim) tensor
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    """Residual block with time and context conditioning."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First conv
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding projection
        self.time_emb_proj = nn.Linear(time_embed_dim, out_channels)
        
        # Second conv
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with time conditioning.
        
        Args:
            x: (batch, in_channels, h, w)
            time_emb: (batch, time_embed_dim)
        
        Returns:
            out: (batch, out_channels, h, w)
        """
        h = x
        
        # First conv
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = self.time_emb_proj(F.silu(time_emb))
        h = h + time_emb[:, :, None, None]
        
        # Second conv
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class SpatialTransformer(nn.Module):
    """Spatial transformer with cross-attention for text conditioning."""
    
    def __init__(
        self,
        channels: int,
        num_heads: int,
        context_dim: int,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        
        # Self-attention
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Cross-attention
        self.norm_cross = nn.LayerNorm(channels)
        self.q_cross = nn.Linear(channels, channels)
        self.kv_cross = nn.Linear(context_dim, channels * 2)
        self.proj_cross = nn.Linear(channels, channels)
        
        # FFN
        self.norm_ffn = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional cross-attention.
        
        Args:
            x: (batch, channels, h, w)
            context: (batch, seq_len, context_dim) text embeddings
        
        Returns:
            out: (batch, channels, h, w)
        """
        B, C, H, W = x.shape
        residual = x
        
        # Self-attention
        h = self.norm(x)
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, num_heads, HW, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        h = (attn @ v).transpose(2, 3).reshape(B, C, H, W)
        h = self.proj_out(h)
        x = x + h
        
        # Cross-attention (if context provided)
        if context is not None:
            h = x.reshape(B, C, H * W).transpose(1, 2)  # (B, HW, C)
            h = self.norm_cross(h)
            
            q = self.q_cross(h)  # (B, HW, C)
            kv = self.kv_cross(context)  # (B, seq_len, 2*C)
            k, v = kv.chunk(2, dim=-1)  # Each (B, seq_len, C)
            
            # Reshape for multi-head
            q = q.reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Attention
            scale = self.head_dim ** -0.5
            attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
            h = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
            h = self.proj_cross(h)
            
            h = h.transpose(1, 2).reshape(B, C, H, W)
            x = x + h
        
        # FFN
        h = x.reshape(B, C, H * W).transpose(1, 2)
        h = self.norm_ffn(h)
        h = self.ffn(h)
        h = h.transpose(1, 2).reshape(B, C, H, W)
        x = x + h
        
        return x


class Downsample(nn.Module):
    """Downsample by factor of 2."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsample by factor of 2."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    U-Net for latent diffusion with text conditioning.
    
    Takes noisy latent and timestep, optionally conditioned on text,
    and predicts the noise.
    """
    
    def __init__(self, config: UNetConfig):
        super().__init__()
        self.config = config
        
        # Time embedding
        self.time_embed = nn.Sequential(
            TimestepEmbedding(config.model_channels),
            nn.Linear(config.model_channels, config.time_embed_dim),
            nn.SiLU(),
            nn.Linear(config.time_embed_dim, config.time_embed_dim),
        )
        
        # Input convolution
        self.conv_in = nn.Conv2d(config.in_channels, config.model_channels, kernel_size=3, padding=1)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        channels = config.model_channels
        input_block_channels = [config.model_channels]
        
        for level, mult in enumerate(config.channel_multipliers):
            out_channels = config.model_channels * mult
            
            for i in range(config.num_res_blocks):
                block = nn.ModuleList([
                    ResBlock(channels, out_channels, config.time_embed_dim, config.dropout)
                ])
                channels = out_channels
                
                # Add attention at specified resolutions
                if level in config.attention_resolutions:
                    block.append(SpatialTransformer(channels, config.num_heads, config.context_dim))
                
                self.down_blocks.append(block)
                input_block_channels.append(channels)
            
            # Downsample (except last level)
            if level < len(config.channel_multipliers) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(channels)]))
                input_block_channels.append(channels)
        
        # Middle
        self.middle_block = nn.ModuleList([
            ResBlock(channels, channels, config.time_embed_dim, config.dropout),
            SpatialTransformer(channels, config.num_heads, config.context_dim),
            ResBlock(channels, channels, config.time_embed_dim, config.dropout),
        ])
        
        # Upsampling path
        self.up_blocks = nn.ModuleList()
        
        for level, mult in enumerate(reversed(config.channel_multipliers)):
            out_channels = config.model_channels * mult
            
            for i in range(config.num_res_blocks + 1):
                # Input channels include skip connection
                skip_channels = input_block_channels.pop()
                channels_in = channels + skip_channels
                
                block = nn.ModuleList([
                    ResBlock(channels_in, out_channels, config.time_embed_dim, config.dropout)
                ])
                channels = out_channels
                
                # Add attention at specified resolutions
                rev_level = len(config.channel_multipliers) - 1 - level
                if rev_level in config.attention_resolutions:
                    block.append(SpatialTransformer(channels, config.num_heads, config.context_dim))
                
                self.up_blocks.append(block)
            
            # Upsample (except last level)
            if level < len(config.channel_multipliers) - 1:
                self.up_blocks.append(nn.ModuleList([Upsample(channels)]))
        
        # Output
        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = nn.Conv2d(channels, config.out_channels, kernel_size=3, padding=1)
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
        Args:
            x: (batch, in_channels, h, w) noisy latent
            timesteps: (batch,) timestep values
            context: (batch, seq_len, context_dim) text embeddings
        
        Returns:
            noise_pred: (batch, out_channels, h, w) predicted noise
        """
        # Time embedding
        time_emb = self.time_embed(timesteps)
        
        # Input
        h = self.conv_in(x)
        
        # Downsampling with skip connections
        skip_connections = [h]
        
        for block in self.down_blocks:
            for layer in block:
                if isinstance(layer, ResBlock):
                    h = layer(h, time_emb)
                elif isinstance(layer, SpatialTransformer):
                    h = layer(h, context)
                elif isinstance(layer, Downsample):
                    h = layer(h)
            
            # Save skip connection after each block
            skip_connections.append(h)
        
        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ResBlock):
                h = layer(h, time_emb)
            elif isinstance(layer, SpatialTransformer):
                h = layer(h, context)
        
        # Upsampling with skip connections
        for block in self.up_blocks:
            for layer in block:
                if isinstance(layer, ResBlock):
                    # Concatenate skip connection
                    h = torch.cat([h, skip_connections.pop()], dim=1)
                    h = layer(h, time_emb)
                elif isinstance(layer, SpatialTransformer):
                    h = layer(h, context)
                elif isinstance(layer, Upsample):
                    h = layer(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
