"""
Text-to-Video diffusion model.

Based on "Video Diffusion Models" (Ho et al., 2022) and "CogVideoX" (Yang et al., 2024).
Generates video directly from text prompts.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict
from tqdm import tqdm

from .video_vae import VideoAutoencoder, VideoVAEConfig
from .scheduler import DDPMScheduler, SchedulerConfig
from .temporal_layers import TemporalResBlock, TemporalAttention
from ..text.transformer import TransformerEncoder, TransformerConfig


@dataclass
class TextToVideoConfig:
    """Configuration for Text-to-Video model."""
    
    # VAE
    vae_config: VideoVAEConfig = None
    
    # Text encoder
    text_encoder_config: TransformerConfig = None
    
    # U-Net (3D)
    in_channels: int = 4
    out_channels: int = 4
    model_channels: int = 320
    channel_multipliers: tuple = (1, 2, 4, 4)
    num_res_blocks: int = 2
    num_heads: int = 8
    context_dim: int = 768
    time_embed_dim: int = 1280
    dropout: float = 0.0
    
    # Scheduler
    scheduler_config: SchedulerConfig = None
    
    # Video params
    num_frames: int = 16
    height: int = 256
    width: int = 256
    
    # Latent scaling
    latent_scale_factor: float = 0.18215
    
    def __post_init__(self):
        if self.vae_config is None:
            self.vae_config = VideoVAEConfig()
        
        if self.text_encoder_config is None:
            self.text_encoder_config = TransformerConfig(
                vocab_size=49408,
                d_model=self.context_dim,
                n_heads=12,
                n_layers=12,
                max_seq_length=77,
            )
        
        if self.scheduler_config is None:
            self.scheduler_config = SchedulerConfig()


class SpatialTransformer3D(nn.Module):
    """Spatial transformer with cross-attention for video."""
    
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
        
        # Self-attention (spatial)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Cross-attention with text
        self.norm_cross = nn.LayerNorm(channels)
        self.q_cross = nn.Linear(channels, channels)
        self.kv_cross = nn.Linear(context_dim, channels * 2)
        self.proj_cross = nn.Linear(channels, channels)
        
        # Temporal attention
        self.temporal_attn = TemporalAttention(channels, num_heads)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, channels, frames, height, width)
            context: (batch, seq_len, context_dim) text embeddings
        
        Returns:
            out: (batch, channels, frames, height, width)
        """
        B, C, F, H, W = x.shape
        
        # Process each frame with spatial attention
        x_frames = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        
        # Self-attention (spatial)
        residual = x_frames
        h = self.norm(x_frames)
        qkv = self.qkv(h)
        qkv = qkv.reshape(B * F, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scale = self.head_dim ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        h = (attn @ v).transpose(2, 3).reshape(B * F, C, H, W)
        h = self.proj_out(h)
        x_frames = residual + h
        
        # Cross-attention with text (if provided)
        if context is not None:
            h = x_frames.reshape(B * F, C, H * W).transpose(1, 2)  # (B*F, H*W, C)
            h = self.norm_cross(h)
            
            q = self.q_cross(h)  # (B*F, H*W, C)
            
            # Expand context for all frames
            context_expanded = context.unsqueeze(1).expand(-1, F, -1, -1).reshape(B * F, -1, context.shape[-1])
            kv = self.kv_cross(context_expanded)
            k, v = kv.chunk(2, dim=-1)
            
            # Reshape for multi-head
            q = q.reshape(B * F, H * W, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.reshape(B * F, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.reshape(B * F, -1, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Attention
            scale = self.head_dim ** -0.5
            attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
            h = (attn @ v).transpose(1, 2).reshape(B * F, H * W, C)
            h = self.proj_cross(h)
            
            h = h.transpose(1, 2).reshape(B * F, C, H, W)
            x_frames = x_frames + h
        
        # Reshape back to video
        x = x_frames.reshape(B, F, C, H, W).permute(0, 2, 1, 3, 4)
        
        # Temporal attention
        x = self.temporal_attn(x)
        
        return x


class TextToVideoUNet(nn.Module):
    """3D U-Net for text-to-video diffusion."""
    
    def __init__(self, config: TextToVideoConfig):
        super().__init__()
        self.config = config
        
        # Time embedding
        from .unet import TimestepEmbedding
        self.time_embed = nn.Sequential(
            TimestepEmbedding(config.model_channels),
            nn.Linear(config.model_channels, config.time_embed_dim),
            nn.SiLU(),
            nn.Linear(config.time_embed_dim, config.time_embed_dim),
        )
        
        # Input conv
        self.conv_in = nn.Conv2d(config.in_channels, config.model_channels, kernel_size=3, padding=1)
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        channels = config.model_channels
        
        for i, mult in enumerate(config.channel_multipliers):
            out_channels = config.model_channels * mult
            
            for _ in range(config.num_res_blocks):
                self.down_blocks.append(nn.ModuleList([
                    TemporalResBlock(channels, out_channels, config.time_embed_dim, config.dropout),
                    SpatialTransformer3D(out_channels, config.num_heads, config.context_dim),
                ]))
                channels = out_channels
            
            if i < len(config.channel_multipliers) - 1:
                self.down_blocks.append(nn.ModuleList([
                    nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
                ]))
        
        # Middle
        self.mid_blocks = nn.ModuleList([
            TemporalResBlock(channels, channels, config.time_embed_dim, config.dropout),
            SpatialTransformer3D(channels, config.num_heads, config.context_dim),
            TemporalResBlock(channels, channels, config.time_embed_dim, config.dropout),
        ])
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        
        for i, mult in enumerate(reversed(config.channel_multipliers)):
            out_channels = config.model_channels * mult
            
            for _ in range(config.num_res_blocks + 1):
                self.up_blocks.append(nn.ModuleList([
                    TemporalResBlock(channels, out_channels, config.time_embed_dim, config.dropout),
                    SpatialTransformer3D(out_channels, config.num_heads, config.context_dim),
                ]))
                channels = out_channels
            
            if i < len(config.channel_multipliers) - 1:
                self.up_blocks.append(nn.ModuleList([
                    nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
                ]))
        
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
        Forward pass.
        
        Args:
            x: (batch, channels, frames, height, width)
            timesteps: (batch,)
            context: (batch, seq_len, context_dim)
        
        Returns:
            noise_pred: (batch, channels, frames, height, width)
        """
        B, C, F, H, W = x.shape
        
        # Time embedding
        time_emb = self.time_embed(timesteps)
        
        # Input (per frame)
        x = x.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        h = self.conv_in(x)
        h = h.reshape(B, F, -1, H, W).permute(0, 2, 1, 3, 4)
        
        # Encoder
        for block in self.down_blocks:
            for layer in block:
                if isinstance(layer, TemporalResBlock):
                    h = layer(h, time_emb)
                elif isinstance(layer, SpatialTransformer3D):
                    h = layer(h, context)
                elif isinstance(layer, nn.Conv2d):
                    B, C, F, H, W = h.shape
                    h = h.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
                    h = layer(h)
                    _, C, H_new, W_new = h.shape
                    h = h.reshape(B, F, C, H_new, W_new).permute(0, 2, 1, 3, 4)
        
        # Middle
        for layer in self.mid_blocks:
            if isinstance(layer, TemporalResBlock):
                h = layer(h, time_emb)
            elif isinstance(layer, SpatialTransformer3D):
                h = layer(h, context)
        
        # Decoder
        for block in self.up_blocks:
            for layer in block:
                if isinstance(layer, TemporalResBlock):
                    h = layer(h, time_emb)
                elif isinstance(layer, SpatialTransformer3D):
                    h = layer(h, context)
                elif isinstance(layer, nn.ConvTranspose2d):
                    B, C, F, H, W = h.shape
                    h = h.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
                    h = layer(h)
                    _, C, H_new, W_new = h.shape
                    h = h.reshape(B, F, C, H_new, W_new).permute(0, 2, 1, 3, 4)
        
        # Output
        B, C, F, H, W = h.shape
        h = h.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        h = self.norm_out(h)
        h = nn.functional.silu(h)
        h = self.conv_out(h)
        h = h.reshape(B, F, self.config.out_channels, H, W).permute(0, 2, 1, 3, 4)
        
        return h


class TextToVideoModel(nn.Module):
    """
    Text-to-Video diffusion model.
    
    Generates video from text prompts.
    """
    
    def __init__(self, config: TextToVideoConfig):
        super().__init__()
        self.config = config
        
        # VAE
        self.vae = VideoAutoencoder(config.vae_config)
        
        # Text encoder
        self.text_embeddings = nn.Embedding(
            config.text_encoder_config.vocab_size,
            config.text_encoder_config.d_model,
        )
        self.text_encoder = TransformerEncoder(config.text_encoder_config)
        
        # U-Net
        self.unet = TextToVideoUNet(config)
        
        # Scheduler
        self.scheduler = DDPMScheduler(config.scheduler_config)
        
        # Latent scaling
        self.latent_scale_factor = config.latent_scale_factor
    
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode text prompt."""
        x = self.text_embeddings(input_ids)
        text_embeddings = self.text_encoder(x, mask=attention_mask)
        return text_embeddings
    
    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video to latent."""
        latent = self.vae.encode(video, sample=False)
        latent = latent * self.latent_scale_factor
        return latent
    
    @torch.no_grad()
    def decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent to video."""
        latents = latents / self.latent_scale_factor
        video = self.vae.decode(latents)
        return video
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generate video from text prompt.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            num_frames: number of frames
            height: video height
            width: video width
            num_inference_steps: denoising steps
            guidance_scale: classifier-free guidance scale
            generator: random generator
        
        Returns:
            video: (batch, 3, num_frames, height, width)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        num_frames = num_frames or self.config.num_frames
        height = height or self.config.height
        width = width or self.config.width
        
        # Encode text
        text_embeddings = self.encode_text(input_ids, attention_mask)
        
        # Classifier-free guidance
        if guidance_scale > 1.0:
            uncond_input_ids = torch.zeros_like(input_ids)
            uncond_embeddings = self.encode_text(uncond_input_ids)
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Initialize random latents
        latent_h = height // 8
        latent_w = width // 8
        latent_f = num_frames // self.config.vae_config.temporal_compression
        
        latents = torch.randn(
            (batch_size, self.config.in_channels, latent_f, latent_h, latent_w),
            generator=generator,
            device=device,
        )
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Denoising loop
        for t in tqdm(self.scheduler.timesteps, desc="Generating video"):
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            
            # Predict noise
            noise_pred = self.unet(
                latent_model_input,
                t.unsqueeze(0).expand(latent_model_input.shape[0]).to(device),
                context=text_embeddings,
            )
            
            # Classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Denoise
            latents, _ = self.scheduler.step(noise_pred, t, latents, eta=0.0, generator=generator)
        
        # Decode to video
        video = self.decode_latent(latents)
        
        return video
    
    def forward(
        self,
        video: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            video: (batch, 3, frames, height, width)
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
        
        Returns:
            Dictionary with loss
        """
        batch_size = video.shape[0]
        device = video.device
        
        # Encode video
        with torch.no_grad():
            video_latent = self.encode_video(video)
        
        # Sample noise and timesteps
        noise = torch.randn_like(video_latent)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )
        
        # Add noise
        noisy_latents = self.scheduler.add_noise(video_latent, noise, timesteps)
        
        # Encode text
        text_embeddings = self.encode_text(input_ids, attention_mask)
        
        # Predict noise
        noise_pred = self.unet(noisy_latents, timesteps, context=text_embeddings)
        
        # Compute loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        return {
            'loss': loss,
            'noise_pred': noise_pred,
            'noise': noise,
        }
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
