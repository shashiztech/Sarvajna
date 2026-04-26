"""
EnCodec: Neural audio codec for high-fidelity audio compression.

Based on "High Fidelity Neural Audio Compression" (Défossez et al., 2022).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class EnCodecConfig:
    """Configuration for EnCodec."""
    
    # Audio params
    sample_rate: int = 24000
    channels: int = 1
    
    # Encoder/Decoder
    encoder_channels: int = 32
    encoder_strides: tuple = (2, 4, 5, 8)  # Total stride = 320
    encoder_dilations: tuple = (1, 3, 9, 27)
    
    # Quantizer
    codebook_size: int = 1024
    num_codebooks: int = 4  # RVQ depth
    codebook_dim: int = 256
    
    # Discriminator
    use_discriminator: bool = True


class ResidualUnit(nn.Module):
    """Residual unit with dilated convolutions."""
    
    def __init__(
        self,
        channels: int,
        dilation: int = 1,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=7,
            dilation=dilation,
            padding=dilation * 3,
        )
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        residual = x
        
        h = F.elu(x)
        h = self.conv1(h)
        h = F.elu(h)
        h = self.conv2(h)
        
        return h + residual


class EncoderBlock(nn.Module):
    """Encoder block with strided convolution and residual units."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dilations: tuple = (1, 3, 9),
    ):
        super().__init__()
        
        # Strided convolution for downsampling
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride // 2,
        )
        
        # Residual units
        self.residuals = nn.ModuleList([
            ResidualUnit(out_channels, dilation=d)
            for d in dilations
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        for res in self.residuals:
            x = res(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block with transposed convolution and residual units."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dilations: tuple = (1, 3, 9),
    ):
        super().__init__()
        
        # Transposed convolution for upsampling
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride // 2,
        )
        
        # Residual units
        self.residuals = nn.ModuleList([
            ResidualUnit(out_channels, dilation=d)
            for d in dilations
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        for res in self.residuals:
            x = res(x)
        return x


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer (RVQ).
    
    Uses multiple codebooks to progressively quantize residuals.
    """
    
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        num_codebooks: int,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        
        # Codebooks
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, dim))
            for _ in range(num_codebooks)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize input using RVQ.
        
        Args:
            x: (batch, dim, time) continuous features
        
        Returns:
            quantized: (batch, dim, time) quantized features
            codes: (batch, num_codebooks, time) discrete codes
        """
        B, D, T = x.shape
        
        # Transpose for easier processing
        x = x.transpose(1, 2)  # (B, T, D)
        
        residual = x
        quantized = torch.zeros_like(x)
        codes_list = []
        
        # Iteratively quantize residuals
        for codebook in self.codebooks:
            # Compute distances
            distances = torch.cdist(residual.reshape(-1, D), codebook)  # (B*T, codebook_size)
            
            # Get nearest codes
            indices = distances.argmin(dim=-1)  # (B*T,)
            codes_list.append(indices.reshape(B, T))
            
            # Quantize
            quantized_residual = codebook[indices].reshape(B, T, D)
            quantized = quantized + quantized_residual
            
            # Update residual
            residual = residual - quantized_residual
        
        # Stack codes
        codes = torch.stack(codes_list, dim=1)  # (B, num_codebooks, T)
        
        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        
        # Transpose back
        quantized = quantized.transpose(1, 2)  # (B, D, T)
        
        return quantized, codes
    
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode codes to continuous features.
        
        Args:
            codes: (batch, num_codebooks, time)
        
        Returns:
            x: (batch, dim, time)
        """
        B, num_codebooks, T = codes.shape
        
        quantized = torch.zeros(B, T, self.dim, device=codes.device)
        
        for i, codebook in enumerate(self.codebooks):
            quantized = quantized + codebook[codes[:, i]]
        
        quantized = quantized.transpose(1, 2)  # (B, D, T)
        
        return quantized


class EnCodec(nn.Module):
    """
    EnCodec: Neural audio codec.
    
    Compresses audio waveforms to discrete codes and reconstructs them.
    """
    
    def __init__(self, config: EnCodecConfig):
        super().__init__()
        self.config = config
        
        # Input convolution
        self.conv_in = nn.Conv1d(config.channels, config.encoder_channels, kernel_size=7, padding=3)
        
        # Encoder
        channels = config.encoder_channels
        self.encoder_blocks = nn.ModuleList()
        
        for stride in config.encoder_strides:
            out_channels = channels * 2
            self.encoder_blocks.append(
                EncoderBlock(channels, out_channels, stride, config.encoder_dilations)
            )
            channels = out_channels
        
        # Quantizer
        self.quantizer = ResidualVectorQuantizer(
            dim=config.codebook_dim,
            codebook_size=config.codebook_size,
            num_codebooks=config.num_codebooks,
        )
        
        # Project to codebook dim
        self.pre_quant = nn.Conv1d(channels, config.codebook_dim, kernel_size=1)
        self.post_quant = nn.Conv1d(config.codebook_dim, channels, kernel_size=1)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        
        for stride in reversed(config.encoder_strides):
            out_channels = channels // 2
            self.decoder_blocks.append(
                DecoderBlock(channels, out_channels, stride, config.encoder_dilations)
            )
            channels = out_channels
        
        # Output convolution
        self.conv_out = nn.Conv1d(channels, config.channels, kernel_size=7, padding=3)
    
    def encode(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio to discrete codes.
        
        Args:
            audio: (batch, channels, time) waveform
        
        Returns:
            codes: (batch, num_codebooks, time_compressed)
            quantized: (batch, codebook_dim, time_compressed)
        """
        # Encode
        h = self.conv_in(audio)
        
        for block in self.encoder_blocks:
            h = block(h)
        
        # Project to codebook dim
        h = self.pre_quant(h)
        
        # Quantize
        quantized, codes = self.quantizer(h)
        
        return codes, quantized
    
    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized features to audio.
        
        Args:
            quantized: (batch, codebook_dim, time_compressed)
        
        Returns:
            audio: (batch, channels, time)
        """
        # Project back
        h = self.post_quant(quantized)
        
        # Decode
        for block in self.decoder_blocks:
            h = block(h)
        
        # Output
        audio = self.conv_out(h)
        
        return audio
    
    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass for training.
        
        Args:
            audio: (batch, channels, time)
        
        Returns:
            reconstruction: (batch, channels, time)
            codes: (batch, num_codebooks, time_compressed)
            quantized: (batch, codebook_dim, time_compressed)
        """
        codes, quantized = self.encode(audio)
        reconstruction = self.decode(quantized)
        
        return reconstruction, codes, quantized
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
