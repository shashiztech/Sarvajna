"""
Image Variational Autoencoder (VAE) for latent compression.

Based on the VAE used in Latent Diffusion Models:
"High-Resolution Image Synthesis with Latent Diffusion Models"
(Rombach et al., 2022) - https://arxiv.org/abs/2112.10752
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class VAEConfig:
    """VAE configuration."""
    
    # Input/output
    in_channels: int = 3
    out_channels: int = 3
    
    # Latent space
    latent_channels: int = 4
    
    # Architecture
    base_channels: int = 128
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4)  # Downsample 4 times
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int, ...] = (16,)  # Apply attention at these resolutions
    dropout: float = 0.0
    
    # Resolution
    resolution: int = 256
    
    # Training
    kl_weight: float = 1e-6  # Weight for KL divergence loss
    

class ResBlock(nn.Module):
    """Residual block with optional attention."""
    
    def __init__(self, in_channels: int, out_channels: int = None, dropout: float = 0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return self.shortcut(x) + h


class AttentionBlock(nn.Module):
    """Self-attention block for spatial features."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        
        # Compute Q, K, V
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        k = k.reshape(B, C, H * W)  # (B, C, HW)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        
        # Scaled dot-product attention
        scale = C ** -0.5
        attn = torch.softmax(q @ k * scale, dim=-1)  # (B, HW, HW)
        
        # Apply attention
        h = (attn @ v).permute(0, 2, 1).reshape(B, C, H, W)
        h = self.proj(h)
        
        return x + h


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


class Encoder(nn.Module):
    """VAE Encoder: image -> latent distribution."""
    
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        
        # Input convolution
        self.conv_in = nn.Conv2d(config.in_channels, config.base_channels, kernel_size=3, padding=1)
        
        # Downsampling blocks
        channels = config.base_channels
        self.down_blocks = nn.ModuleList()
        
        for i, mult in enumerate(config.channel_multipliers):
            out_channels = config.base_channels * mult
            
            # Residual blocks with channel transition on first block
            for j in range(config.num_res_blocks):
                in_ch = channels if j == 0 else out_channels
                block = ResBlock(in_ch, out_channels, config.dropout)
                self.down_blocks.append(block)
            
            channels = out_channels
            
            # Attention at certain resolutions
            current_res = config.resolution // (2 ** i)
            if current_res in config.attn_resolutions:
                self.down_blocks.append(AttentionBlock(channels))
            
            # Downsample (except last)
            if i < len(config.channel_multipliers) - 1:
                self.down_blocks.append(Downsample(channels))
        
        # Middle blocks
        self.mid_block1 = ResBlock(channels, config.dropout)
        self.mid_attn = AttentionBlock(channels)
        self.mid_block2 = ResBlock(channels, config.dropout)
        
        # Output: mean and logvar for reparameterization
        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = nn.Conv2d(channels, 2 * config.latent_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image to latent distribution.
        
        Args:
            x: (batch_size, in_channels, height, width)
        
        Returns:
            mean: (batch_size, latent_channels, h, w)
            logvar: (batch_size, latent_channels, h, w)
        """
        h = self.conv_in(x)
        
        # Downsample
        for block in self.down_blocks:
            h = block(h)
        
        # Middle
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        # Split into mean and logvar
        mean, logvar = h.chunk(2, dim=1)
        
        return mean, logvar


class Decoder(nn.Module):
    """VAE Decoder: latent -> reconstructed image."""
    
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        
        # Calculate channel progression (reverse of encoder)
        channel_mults_reversed = list(reversed(config.channel_multipliers))
        channels = config.base_channels * channel_mults_reversed[0]
        
        # Input convolution
        self.conv_in = nn.Conv2d(config.latent_channels, channels, kernel_size=3, padding=1)
        
        # Middle blocks
        self.mid_block1 = ResBlock(channels, channels, config.dropout)
        self.mid_attn = AttentionBlock(channels)
        self.mid_block2 = ResBlock(channels, channels, config.dropout)
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        
        for i, mult in enumerate(channel_mults_reversed):
            out_channels = config.base_channels * mult
            
            # Residual blocks with channel transition
            for j in range(config.num_res_blocks + 1):
                in_ch = channels if j == 0 else out_channels
                block = ResBlock(in_ch, out_channels, config.dropout)
                self.up_blocks.append(block)
            
            channels = out_channels
            
            # Attention at certain resolutions
            current_res = config.resolution // (2 ** (len(channel_mults_reversed) - i - 1))
            if current_res in config.attn_resolutions:
                self.up_blocks.append(AttentionBlock(channels))
            
            # Upsample (except last)
            if i < len(channel_mults_reversed) - 1:
                self.up_blocks.append(Upsample(channels))
        
        # Output
        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = nn.Conv2d(channels, config.out_channels, kernel_size=3, padding=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image.
        
        Args:
            z: (batch_size, latent_channels, h, w)
        
        Returns:
            x: (batch_size, out_channels, height, width)
        """
        h = self.conv_in(z)
        
        # Middle
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        
        # Upsample
        for block in self.up_blocks:
            h = block(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h


class ImageAutoencoder(nn.Module):
    """
    Variational Autoencoder for image compression.
    
    Compresses images into a lower-dimensional latent space with
    KL divergence regularization.
    """
    
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
    
    def encode(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Encode image to latent space.
        
        Args:
            x: (batch_size, in_channels, height, width)
            sample: If True, sample from distribution; if False, return mean
        
        Returns:
            z: (batch_size, latent_channels, h, w)
        """
        mean, logvar = self.encoder(x)
        
        if sample:
            # Reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + eps * std
        else:
            z = mean
        
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image.
        
        Args:
            z: (batch_size, latent_channels, h, w)
        
        Returns:
            x: (batch_size, out_channels, height, width)
        """
        return self.decoder(z)
    
    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: (batch_size, in_channels, height, width)
            return_latent: If True, return latent in output dict
        
        Returns:
            Dictionary with:
                - reconstruction: reconstructed image
                - mean: latent mean
                - logvar: latent log variance
                - kl_loss: KL divergence loss
                - recon_loss: reconstruction loss
                - loss: total loss
                - latent: (optional) sampled latent
        """
        # Encode
        mean, logvar = self.encoder(x)
        
        # Sample latent
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        
        # Decode
        recon = self.decoder(z)
        
        # Compute losses
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        
        # Total loss
        loss = recon_loss + self.config.kl_weight * kl_loss
        
        output = {
            'reconstruction': recon,
            'mean': mean,
            'logvar': logvar,
            'kl_loss': kl_loss,
            'recon_loss': recon_loss,
            'loss': loss,
        }
        
        if return_latent:
            output['latent'] = z
        
        return output
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
