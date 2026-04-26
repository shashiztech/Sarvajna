"""
Latent Diffusion Model for text-to-image generation.

Based on "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, List
from tqdm import tqdm

from .unet import UNet, UNetConfig
from .scheduler import DDPMScheduler, SchedulerConfig
from .image_autoencoder import ImageAutoencoder, VAEConfig
from ..text.transformer import TransformerEncoder, TransformerConfig


@dataclass
class LatentDiffusionConfig:
    """Configuration for Latent Diffusion Model."""
    
    # Model components
    unet_config: UNetConfig = None
    vae_config: VAEConfig = None
    text_encoder_config: TransformerConfig = None
    scheduler_config: SchedulerConfig = None
    
    # Latent space
    latent_channels: int = 4
    latent_scale_factor: float = 0.18215  # Standard SD scaling
    
    # Conditioning
    conditioning_key: str = "crossattn"  # "crossattn" or "concat"
    
    def __post_init__(self):
        if self.unet_config is None:
            self.unet_config = UNetConfig(
                in_channels=self.latent_channels,
                out_channels=self.latent_channels,
            )
        
        if self.vae_config is None:
            self.vae_config = VAEConfig(
                latent_channels=self.latent_channels,
            )
        
        if self.text_encoder_config is None:
            self.text_encoder_config = TransformerConfig(
                vocab_size=49408,  # CLIP vocab size
                d_model=768,
                n_heads=12,
                n_layers=12,
                max_seq_length=77,
            )
        
        if self.scheduler_config is None:
            self.scheduler_config = SchedulerConfig()


class LatentDiffusionModel(nn.Module):
    """
    Latent Diffusion Model for text-to-image generation.
    
    Combines:
    - VAE for image encoding/decoding
    - Text encoder for prompt conditioning
    - U-Net for denoising in latent space
    - DDPM/DDIM scheduler for sampling
    """
    
    def __init__(self, config: LatentDiffusionConfig):
        super().__init__()
        self.config = config
        
        # VAE for latent compression
        self.vae = ImageAutoencoder(config.vae_config)
        
        # Text encoder
        self.text_embeddings = nn.Embedding(
            config.text_encoder_config.vocab_size,
            config.text_encoder_config.d_model,
        )
        self.text_encoder = TransformerEncoder(config.text_encoder_config)
        
        # U-Net for denoising
        self.unet = UNet(config.unet_config)
        
        # Noise scheduler
        self.scheduler = DDPMScheduler(config.scheduler_config)
        
        # Latent scaling
        self.latent_scale_factor = config.latent_scale_factor
    
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode text prompt to conditioning embeddings.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
        
        Returns:
            text_embeddings: (batch, seq_len, d_model)
        """
        # Embed tokens
        x = self.text_embeddings(input_ids)
        
        # Encode
        text_embeddings = self.text_encoder(x, mask=attention_mask)
        
        return text_embeddings
    
    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space.
        
        Args:
            images: (batch, 3, height, width) in [-1, 1]
        
        Returns:
            latents: (batch, latent_channels, h, w)
        """
        latents = self.vae.encode(images, sample=False)
        latents = latents * self.latent_scale_factor
        return latents
    
    @torch.no_grad()
    def decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to images.
        
        Args:
            latents: (batch, latent_channels, h, w)
        
        Returns:
            images: (batch, 3, height, width) in [-1, 1]
        """
        latents = latents / self.latent_scale_factor
        images = self.vae.decode(latents)
        return images
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            images: (batch, 3, height, width) in [-1, 1]
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
        
        Returns:
            Dictionary with loss and other outputs
        """
        batch_size = images.shape[0]
        device = images.device
        
        # Encode image to latent
        with torch.no_grad():
            latents = self.encode_image(images)
        
        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )
        
        # Add noise to latents
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # Get text conditioning
        text_embeddings = self.encode_text(input_ids, attention_mask)
        
        # Predict noise
        noise_pred = self.unet(noisy_latents, timesteps, context=text_embeddings)
        
        # Compute loss (simple MSE)
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        return {
            'loss': loss,
            'noise_pred': noise_pred,
            'noise': noise,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate images from text prompts.
        
        Args:
            input_ids: (batch, seq_len) text prompts
            attention_mask: (batch, seq_len)
            height: output image height
            width: output image width
            num_inference_steps: number of denoising steps
            guidance_scale: classifier-free guidance scale (1.0 = no guidance)
            eta: DDIM eta parameter (0.0 = deterministic)
            generator: random generator
            latents: optional starting latents
        
        Returns:
            images: (batch, 3, height, width) in [-1, 1]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Set inference timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Encode conditional prompt
        text_embeddings = self.encode_text(input_ids, attention_mask)
        
        # Classifier-free guidance: encode unconditional prompt
        if guidance_scale > 1.0:
            uncond_input_ids = torch.zeros_like(input_ids)
            uncond_embeddings = self.encode_text(uncond_input_ids)
            
            # Concatenate for batch processing
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Prepare latents
        if latents is None:
            latent_h = height // 8  # VAE downsamples by 8
            latent_w = width // 8
            latents = torch.randn(
                (batch_size, self.config.latent_channels, latent_h, latent_w),
                generator=generator,
                device=device,
            )
        
        # Denoising loop
        for t in tqdm(self.scheduler.timesteps, desc="Generating"):
            # Expand latents for classifier-free guidance
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
            latents, _ = self.scheduler.step(noise_pred, t, latents, eta=eta, generator=generator)
        
        # Decode latents to images
        images = self.decode_latent(latents)
        
        return images
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


# Helper functions
def latent_diffusion_base() -> LatentDiffusionModel:
    """Create base Latent Diffusion Model."""
    config = LatentDiffusionConfig()
    return LatentDiffusionModel(config)


def latent_diffusion_large() -> LatentDiffusionModel:
    """Create large Latent Diffusion Model with more capacity."""
    unet_config = UNetConfig(
        model_channels=640,
        channel_multipliers=(1, 2, 4, 4),
        num_heads=16,
    )
    
    config = LatentDiffusionConfig(unet_config=unet_config)
    return LatentDiffusionModel(config)
