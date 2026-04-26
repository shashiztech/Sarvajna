"""
Image-to-Video diffusion model.

Based on "Stable Video Diffusion" (Blattmann et al., 2023).
Generates video from a single image frame.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict
from tqdm import tqdm

from .video_vae import VideoAutoencoder, VideoVAEConfig
from .scheduler import DDPMScheduler, SchedulerConfig
from .temporal_layers import TemporalResBlock, TemporalAttention


@dataclass
class ImageToVideoConfig:
    """Configuration for Image-to-Video model."""
    
    # VAE
    vae_config: VideoVAEConfig = None
    
    # U-Net (3D extension)
    in_channels: int = 4
    out_channels: int = 4
    model_channels: int = 320
    channel_multipliers: tuple = (1, 2, 4, 4)
    num_res_blocks: int = 2
    num_heads: int = 8
    time_embed_dim: int = 1280
    dropout: float = 0.0
    
    # Scheduler
    scheduler_config: SchedulerConfig = None
    
    # Conditioning
    num_frames: int = 14  # Number of frames to generate
    fps: int = 7  # Frames per second
    motion_bucket_id: int = 127  # Motion intensity (0-255)
    
    # Latent scaling
    latent_scale_factor: float = 0.18215
    
    def __post_init__(self):
        if self.vae_config is None:
            self.vae_config = VideoVAEConfig()
        
        if self.scheduler_config is None:
            self.scheduler_config = SchedulerConfig()


class VideoUNet(nn.Module):
    """
    3D U-Net for video diffusion.
    
    Extends 2D U-Net with temporal layers.
    """
    
    def __init__(self, config: ImageToVideoConfig):
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
        
        # FPS embedding
        self.fps_embed = nn.Embedding(30, config.time_embed_dim)  # Support up to 30 FPS
        
        # Motion embedding
        self.motion_embed = nn.Embedding(256, config.time_embed_dim)  # 0-255 motion bucket
        
        # Input conv (2D, applied per frame)
        self.conv_in = nn.Conv2d(config.in_channels + config.in_channels, config.model_channels, kernel_size=3, padding=1)
        
        # Build encoder
        self.down_blocks = nn.ModuleList()
        channels = config.model_channels
        
        for i, mult in enumerate(config.channel_multipliers):
            out_channels = config.model_channels * mult
            
            for _ in range(config.num_res_blocks):
                self.down_blocks.append(nn.ModuleList([
                    TemporalResBlock(channels, out_channels, config.time_embed_dim, config.dropout),
                    TemporalAttention(out_channels, config.num_heads),
                ]))
                channels = out_channels
            
            # Downsample
            if i < len(config.channel_multipliers) - 1:
                self.down_blocks.append(nn.ModuleList([
                    nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
                ]))
        
        # Middle
        self.mid_blocks = nn.ModuleList([
            TemporalResBlock(channels, channels, config.time_embed_dim, config.dropout),
            TemporalAttention(channels, config.num_heads),
            TemporalResBlock(channels, channels, config.time_embed_dim, config.dropout),
        ])
        
        # Build decoder
        self.up_blocks = nn.ModuleList()
        
        for i, mult in enumerate(reversed(config.channel_multipliers)):
            out_channels = config.model_channels * mult
            
            for _ in range(config.num_res_blocks + 1):
                self.up_blocks.append(nn.ModuleList([
                    TemporalResBlock(channels, out_channels, config.time_embed_dim, config.dropout),
                    TemporalAttention(out_channels, config.num_heads),
                ]))
                channels = out_channels
            
            # Upsample
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
        image_cond: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        motion_bucket_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, channels, frames, height, width) noisy latent video
            timesteps: (batch,) diffusion timesteps
            image_cond: (batch, channels, 1, height, width) conditioning image latent
            fps: (batch,) frames per second
            motion_bucket_id: (batch,) motion intensity
        
        Returns:
            noise_pred: (batch, channels, frames, height, width)
        """
        B, C, F, H, W = x.shape
        device = x.device
        
        # Time embedding
        time_emb = self.time_embed(timesteps)
        
        # FPS conditioning
        if fps is not None:
            fps_emb = self.fps_embed(fps)
            time_emb = time_emb + fps_emb
        
        # Motion conditioning
        if motion_bucket_id is not None:
            motion_emb = self.motion_embed(motion_bucket_id)
            time_emb = time_emb + motion_emb
        
        # Concatenate image conditioning to each frame
        image_cond = image_cond.expand(-1, -1, F, -1, -1)  # (B, C, F, H, W)
        x_cat = torch.cat([x, image_cond], dim=1)  # (B, 2*C, F, H, W)
        
        # Process per frame through input conv
        x_cat = x_cat.permute(0, 2, 1, 3, 4).reshape(B * F, 2 * C, H, W)
        h = self.conv_in(x_cat)
        h = h.reshape(B, F, self.config.model_channels, H, W).permute(0, 2, 1, 3, 4)
        
        # Encoder
        for block in self.down_blocks:
            for layer in block:
                if isinstance(layer, TemporalResBlock):
                    h = layer(h, time_emb)
                elif isinstance(layer, TemporalAttention):
                    h = layer(h)
                elif isinstance(layer, nn.Conv2d):
                    # Downsample per frame
                    B, C, F, H, W = h.shape
                    h = h.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
                    h = layer(h)
                    _, C, H_new, W_new = h.shape
                    h = h.reshape(B, F, C, H_new, W_new).permute(0, 2, 1, 3, 4)
        
        # Middle
        for layer in self.mid_blocks:
            if isinstance(layer, TemporalResBlock):
                h = layer(h, time_emb)
            elif isinstance(layer, TemporalAttention):
                h = layer(h)
        
        # Decoder
        for block in self.up_blocks:
            for layer in block:
                if isinstance(layer, TemporalResBlock):
                    h = layer(h, time_emb)
                elif isinstance(layer, TemporalAttention):
                    h = layer(h)
                elif isinstance(layer, nn.ConvTranspose2d):
                    # Upsample per frame
                    B, C, F, H, W = h.shape
                    h = h.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
                    h = layer(h)
                    _, C, H_new, W_new = h.shape
                    h = h.reshape(B, F, C, H_new, W_new).permute(0, 2, 1, 3, 4)
        
        # Output per frame
        B, C, F, H, W = h.shape
        h = h.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        h = self.norm_out(h)
        h = nn.functional.silu(h)
        h = self.conv_out(h)
        h = h.reshape(B, F, self.config.out_channels, H, W).permute(0, 2, 1, 3, 4)
        
        return h


class ImageToVideoModel(nn.Module):
    """
    Image-to-Video diffusion model.
    
    Generates video from a single image frame.
    """
    
    def __init__(self, config: ImageToVideoConfig):
        super().__init__()
        self.config = config
        
        # VAE
        self.vae = VideoAutoencoder(config.vae_config)
        
        # U-Net
        self.unet = VideoUNet(config)
        
        # Scheduler
        self.scheduler = DDPMScheduler(config.scheduler_config)
        
        # Latent scaling
        self.latent_scale_factor = config.latent_scale_factor
    
    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent (single frame)."""
        # Add temporal dimension
        image = image.unsqueeze(2)  # (B, 3, 1, H, W)
        latent = self.vae.encode(image, sample=False)
        latent = latent * self.latent_scale_factor
        return latent
    
    @torch.no_grad()
    def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode video latents to frames."""
        latents = latents / self.latent_scale_factor
        video = self.vae.decode(latents)
        return video
    
    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generate video from image.
        
        Args:
            image: (batch, 3, height, width) conditioning image in [-1, 1]
            num_frames: number of frames to generate
            num_inference_steps: denoising steps
            fps: frames per second
            motion_bucket_id: motion intensity (0-255)
            noise_aug_strength: noise augmentation for conditioning
            generator: random generator
        
        Returns:
            video: (batch, 3, num_frames, height, width) in [-1, 1]
        """
        batch_size = image.shape[0]
        device = image.device
        num_frames = num_frames or self.config.num_frames
        
        # Encode image to latent
        image_latent = self.encode_image(image)  # (B, C, 1, H, W)
        
        # Add noise augmentation
        if noise_aug_strength > 0:
            noise = torch.randn_like(image_latent) * noise_aug_strength
            image_latent = image_latent + noise
        
        # Prepare conditioning
        fps_tensor = torch.tensor([fps] * batch_size, device=device, dtype=torch.long)
        motion_tensor = torch.tensor([motion_bucket_id] * batch_size, device=device, dtype=torch.long)
        
        # Initialize random latents
        _, C, _, H, W = image_latent.shape
        latents = torch.randn(
            (batch_size, C, num_frames, H, W),
            generator=generator,
            device=device,
        )
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Denoising loop
        for t in tqdm(self.scheduler.timesteps, desc="Generating video"):
            # Predict noise
            noise_pred = self.unet(
                latents,
                t.unsqueeze(0).expand(batch_size).to(device),
                image_latent,
                fps_tensor,
                motion_tensor,
            )
            
            # Denoise
            latents, _ = self.scheduler.step(noise_pred, t, latents, eta=0.0, generator=generator)
        
        # Decode to video
        video = self.decode_video(latents)
        
        return video
    
    def forward(
        self,
        video: torch.Tensor,
        image: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        motion_bucket_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            video: (batch, 3, frames, height, width) target video
            image: (batch, 3, height, width) conditioning image (first frame)
            fps: (batch,) frames per second
            motion_bucket_id: (batch,) motion intensity
        
        Returns:
            Dictionary with loss and outputs
        """
        batch_size = video.shape[0]
        device = video.device
        
        # Encode video and image
        with torch.no_grad():
            video_latent = self.vae.encode(video, sample=False)
            video_latent = video_latent * self.latent_scale_factor
            
            image_latent = self.encode_image(image)
        
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
        
        # Predict noise
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            image_latent,
            fps,
            motion_bucket_id,
        )
        
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
