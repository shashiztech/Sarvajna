"""
Noise schedulers for diffusion models.

Implements DDPM and DDIM sampling strategies.
Based on:
- "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- "Denoising Diffusion Implicit Models" (Song et al., 2021)
"""

import torch
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class NoiseScheduleType(Enum):
    """Types of noise schedules."""
    LINEAR = "linear"
    COSINE = "cosine"
    SCALED_LINEAR = "scaled_linear"


@dataclass
class SchedulerConfig:
    """Configuration for noise scheduler."""
    
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    schedule_type: NoiseScheduleType = NoiseScheduleType.SCALED_LINEAR
    
    # Sampling
    num_inference_steps: int = 50
    eta: float = 0.0  # 0.0 = DDIM, 1.0 = DDPM


class DDPMScheduler:
    """
    DDPM/DDIM noise scheduler for diffusion models.
    
    Handles forward diffusion (adding noise) and reverse diffusion (denoising).
    """
    
    def __init__(self, config: SchedulerConfig):
        self.config = config
        
        # Compute beta schedule
        if config.schedule_type == NoiseScheduleType.LINEAR:
            betas = np.linspace(config.beta_start, config.beta_end, config.num_train_timesteps)
        elif config.schedule_type == NoiseScheduleType.SCALED_LINEAR:
            # Stable Diffusion uses scaled linear
            betas = np.linspace(
                config.beta_start ** 0.5,
                config.beta_end ** 0.5,
                config.num_train_timesteps
            ) ** 2
        elif config.schedule_type == NoiseScheduleType.COSINE:
            # Cosine schedule
            timesteps = np.arange(config.num_train_timesteps + 1)
            alphas_cumprod = np.cos(((timesteps / config.num_train_timesteps) + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = np.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {config.schedule_type}")
        
        self.betas = torch.from_numpy(betas).float()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        
        # Timesteps for inference
        self.set_timesteps(config.num_inference_steps)
    
    def set_timesteps(self, num_inference_steps: int):
        """Set timesteps for inference sampling."""
        self.num_inference_steps = num_inference_steps
        
        # Linear spacing in timestep space
        self.timesteps = torch.linspace(
            self.config.num_train_timesteps - 1,
            0,
            num_inference_steps,
            dtype=torch.long
        )
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward diffusion: add noise to clean samples.
        
        q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        
        Args:
            original_samples: (batch, channels, h, w) clean samples
            noise: (batch, channels, h, w) Gaussian noise
            timesteps: (batch,) timestep values
        
        Returns:
            noisy_samples: (batch, channels, h, w)
        """
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(original_samples.device)[timesteps]
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(original_samples.device)[timesteps]
        
        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod.shape) < len(original_samples.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.unsqueeze(-1)
        
        noisy_samples = (
            sqrt_alphas_cumprod * original_samples +
            sqrt_one_minus_alphas_cumprod * noise
        )
        
        return noisy_samples
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reverse diffusion step: denoise sample.
        
        Args:
            model_output: (batch, channels, h, w) predicted noise
            timestep: current timestep
            sample: (batch, channels, h, w) current noisy sample
            eta: DDIM eta parameter (0.0 = deterministic, 1.0 = DDPM)
            generator: random generator for stochastic sampling
        
        Returns:
            prev_sample: (batch, channels, h, w) denoised sample at t-1
            pred_original_sample: (batch, channels, h, w) predicted x_0
        """
        if eta is None:
            eta = self.config.eta
        
        # Get the previous timestep
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
        
        # Get alpha values
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        
        # Predict original sample from noise
        pred_original_sample = (
            sample - torch.sqrt(beta_prod_t) * model_output
        ) / torch.sqrt(alpha_prod_t)
        
        # Clip predicted x_0
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # Compute variance
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * torch.sqrt(variance)
        
        # Compute "direction pointing to x_t"
        pred_sample_direction = torch.sqrt(1 - alpha_prod_t_prev - std_dev_t**2) * model_output
        
        # Compute previous sample mean
        prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        
        # Add noise if eta > 0 (DDPM)
        if eta > 0:
            noise = torch.randn(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample + std_dev_t * noise
        
        return prev_sample, pred_original_sample
    
    def _get_variance(self, timestep: int, prev_timestep: int) -> torch.Tensor:
        """Compute variance for timestep."""
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        
        return variance
    
    def __len__(self) -> int:
        """Return number of timesteps."""
        return self.config.num_train_timesteps


def make_ddpm_schedule(
    num_timesteps: int = 1000,
    beta_start: float = 0.00085,
    beta_end: float = 0.012,
    schedule_type: str = "scaled_linear",
) -> DDPMScheduler:
    """
    Helper to create DDPM scheduler with default settings.
    
    Args:
        num_timesteps: Number of diffusion timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
        schedule_type: "linear", "scaled_linear", or "cosine"
    
    Returns:
        DDPMScheduler instance
    """
    config = SchedulerConfig(
        num_train_timesteps=num_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        schedule_type=NoiseScheduleType(schedule_type),
    )
    
    return DDPMScheduler(config)


def make_ddim_schedule(
    num_timesteps: int = 1000,
    num_inference_steps: int = 50,
    beta_start: float = 0.00085,
    beta_end: float = 0.012,
) -> DDPMScheduler:
    """
    Helper to create DDIM scheduler (deterministic sampling).
    
    Args:
        num_timesteps: Number of training diffusion timesteps
        num_inference_steps: Number of inference steps (can be much smaller)
        beta_start: Starting beta value
        beta_end: Ending beta value
    
    Returns:
        DDPMScheduler configured for DDIM
    """
    config = SchedulerConfig(
        num_train_timesteps=num_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        schedule_type=NoiseScheduleType.SCALED_LINEAR,
        num_inference_steps=num_inference_steps,
        eta=0.0,  # Deterministic
    )
    
    return DDPMScheduler(config)
