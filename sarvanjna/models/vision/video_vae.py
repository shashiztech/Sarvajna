"""
Video VAE for latent video compression.

Based on "Stable Video Diffusion" (Blattmann et al., 2023).
Extends the image autoencoder to handle temporal dimension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from .image_autoencoder import ResBlock, AttentionBlock


@dataclass
class VideoVAEConfig:
    """Configuration for Video VAE."""
    
    in_channels: int = 3
    base_channels: int = 128
    channel_multipliers: tuple = (1, 2, 4, 4)
    latent_channels: int = 4
    num_res_blocks: int = 2
    
    # Temporal compression
    temporal_compression: int = 4  # Compress time by this factor


class TemporalDownsample(nn.Module):
    """Downsample temporal dimension."""
    
    def __init__(self, channels: int, factor: int = 2):
        super().__init__()
        self.factor = factor
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(factor, 1, 1),
            stride=(factor, 1, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, frames, height, width)
        Returns:
            out: (batch, channels, frames//factor, height, width)
        """
        return self.conv(x)


class TemporalUpsample(nn.Module):
    """Upsample temporal dimension."""
    
    def __init__(self, channels: int, factor: int = 2):
        super().__init__()
        self.factor = factor
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, frames, height, width)
        Returns:
            out: (batch, channels, frames*factor, height, width)
        """
        # Interpolate temporally
        B, C, F, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, F, C, H, W)
        x = x.reshape(B * F, C, H, W)
        
        # Repeat frames
        x = x.repeat_interleave(self.factor, dim=0)  # (B*F*factor, C, H, W)
        x = x.reshape(B, F * self.factor, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, F*factor, H, W)
        
        # Apply conv
        x = self.conv(x)
        
        return x


class VideoResBlock3D(nn.Module):
    """3D Residual block for video."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, frames, height, width)
        Returns:
            out: (batch, out_channels, frames, height, width)
        """
        residual = x
        
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + self.shortcut(residual)


class VideoEncoder(nn.Module):
    """Video encoder with temporal compression."""
    
    def __init__(self, config: VideoVAEConfig):
        super().__init__()
        self.config = config
        
        # Input conv
        self.conv_in = nn.Conv3d(config.in_channels, config.base_channels, kernel_size=3, padding=1)
        
        # Downsampling blocks
        channels = config.base_channels
        self.down_blocks = nn.ModuleList()
        
        for i, mult in enumerate(config.channel_multipliers):
            out_channels = config.base_channels * mult
            
            # Res blocks
            for _ in range(config.num_res_blocks):
                self.down_blocks.append(VideoResBlock3D(channels, out_channels))
                channels = out_channels
            
            # Spatial downsample
            if i < len(config.channel_multipliers) - 1:
                self.down_blocks.append(
                    nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
                )
        
        # Temporal downsample
        self.temporal_down = TemporalDownsample(channels, factor=config.temporal_compression)
        
        # Middle
        self.mid_block1 = VideoResBlock3D(channels, channels)
        self.mid_attn = AttentionBlock(channels)  # Spatial attention
        self.mid_block2 = VideoResBlock3D(channels, channels)
        
        # Output (to latent)
        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = nn.Conv3d(channels, config.latent_channels * 2, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Encode video to latent distribution.
        
        Args:
            x: (batch, 3, frames, height, width)
        
        Returns:
            mean, logvar: (batch, latent_channels, frames//temporal_compression, h, w)
        """
        B, C, F, H, W = x.shape
        
        # Input
        h = self.conv_in(x)
        
        # Downsample
        for block in self.down_blocks:
            h = block(h)
        
        # Temporal compression
        h = self.temporal_down(h)
        
        # Middle (apply attention per frame)
        h = self.mid_block1(h)
        
        # Spatial attention (process each frame)
        B, C, F_compressed, H_down, W_down = h.shape
        h = h.permute(0, 2, 1, 3, 4).reshape(B * F_compressed, C, H_down, W_down)
        h = self.mid_attn(h)
        h = h.reshape(B, F_compressed, C, H_down, W_down).permute(0, 2, 1, 3, 4)
        
        h = self.mid_block2(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        # Split into mean and logvar
        mean, logvar = h.chunk(2, dim=1)
        
        return mean, logvar


class VideoDecoder(nn.Module):
    """Video decoder with temporal upsampling."""
    
    def __init__(self, config: VideoVAEConfig):
        super().__init__()
        self.config = config
        
        # Latent channels
        channels = config.base_channels * config.channel_multipliers[-1]
        
        # Input conv
        self.conv_in = nn.Conv3d(config.latent_channels, channels, kernel_size=3, padding=1)
        
        # Middle
        self.mid_block1 = VideoResBlock3D(channels, channels)
        self.mid_attn = AttentionBlock(channels)
        self.mid_block2 = VideoResBlock3D(channels, channels)
        
        # Temporal upsample
        self.temporal_up = TemporalUpsample(channels, factor=config.temporal_compression)
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        
        for i, mult in enumerate(reversed(config.channel_multipliers)):
            out_channels = config.base_channels * mult
            
            # Res blocks
            for _ in range(config.num_res_blocks + 1):
                self.up_blocks.append(VideoResBlock3D(channels, out_channels))
                channels = out_channels
            
            # Spatial upsample
            if i < len(config.channel_multipliers) - 1:
                self.up_blocks.append(
                    nn.ConvTranspose3d(
                        channels,
                        channels,
                        kernel_size=(1, 4, 4),
                        stride=(1, 2, 2),
                        padding=(0, 1, 1),
                    )
                )
        
        # Output
        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = nn.Conv3d(channels, config.in_channels, kernel_size=3, padding=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to video.
        
        Args:
            z: (batch, latent_channels, frames_compressed, h, w)
        
        Returns:
            video: (batch, 3, frames, height, width)
        """
        # Input
        h = self.conv_in(z)
        
        # Middle
        h = self.mid_block1(h)
        
        # Spatial attention
        B, C, F, H, W = h.shape
        h = h.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        h = self.mid_attn(h)
        h = h.reshape(B, F, C, H, W).permute(0, 2, 1, 3, 4)
        
        h = self.mid_block2(h)
        
        # Temporal upsample
        h = self.temporal_up(h)
        
        # Upsample
        for block in self.up_blocks:
            h = block(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h


class VideoAutoencoder(nn.Module):
    """
    Video VAE for latent video compression.
    
    Compresses video both spatially and temporally.
    """
    
    def __init__(self, config: VideoVAEConfig):
        super().__init__()
        self.config = config
        
        self.encoder = VideoEncoder(config)
        self.decoder = VideoDecoder(config)
    
    def encode(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Encode video to latent.
        
        Args:
            x: (batch, 3, frames, height, width) in [-1, 1]
            sample: whether to sample from distribution
        
        Returns:
            z: (batch, latent_channels, frames//compression, h, w)
        """
        mean, logvar = self.encoder(x)
        
        if sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + eps * std
        else:
            z = mean
        
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to video.
        
        Args:
            z: (batch, latent_channels, frames_compressed, h, w)
        
        Returns:
            video: (batch, 3, frames, height, width) in [-1, 1]
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Full forward pass for training.
        
        Args:
            x: (batch, 3, frames, height, width)
        
        Returns:
            reconstruction, mean, logvar
        """
        mean, logvar = self.encoder(x)
        
        # Reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        
        # Decode
        reconstruction = self.decoder(z)
        
        return reconstruction, mean, logvar
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
