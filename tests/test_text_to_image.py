"""
Unit tests for text-to-image models (Latent Diffusion).
"""

import unittest
import torch

from sarvanjna.models.vision import (
    UNet,
    UNetConfig,
    DDPMScheduler,
    SchedulerConfig,
    NoiseScheduleType,
    LatentDiffusionModel,
    LatentDiffusionConfig,
)


class TestUNet(unittest.TestCase):
    """Test U-Net model."""
    
    def setUp(self):
        self.config = UNetConfig(
            in_channels=4,
            out_channels=4,
            model_channels=64,  # Small for testing
            channel_multipliers=(1, 2, 4),
            num_res_blocks=1,
            attention_resolutions=(2, 1),
            num_heads=4,
            context_dim=128,
        )
        self.model = UNet(self.config)
        self.model.eval()
    
    def test_forward_unconditional(self):
        """Test forward pass without text conditioning."""
        batch_size = 2
        h, w = 32, 32
        
        x = torch.randn(batch_size, 4, h, w)
        timesteps = torch.randint(0, 1000, (batch_size,))
        
        output = self.model(x, timesteps)
        
        self.assertEqual(output.shape, (batch_size, 4, h, w))
    
    def test_forward_conditional(self):
        """Test forward pass with text conditioning."""
        batch_size = 2
        h, w = 32, 32
        seq_len = 16
        
        x = torch.randn(batch_size, 4, h, w)
        timesteps = torch.randint(0, 1000, (batch_size,))
        context = torch.randn(batch_size, seq_len, 128)
        
        output = self.model(x, timesteps, context)
        
        self.assertEqual(output.shape, (batch_size, 4, h, w))
    
    def test_gradient_flow(self):
        """Test gradient flow through model."""
        x = torch.randn(1, 4, 32, 32, requires_grad=True)
        timesteps = torch.tensor([500])
        context = torch.randn(1, 16, 128)
        
        output = self.model(x, timesteps, context)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)


class TestDDPMScheduler(unittest.TestCase):
    """Test DDPM scheduler."""
    
    def setUp(self):
        self.config = SchedulerConfig(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            num_inference_steps=50,
        )
        self.scheduler = DDPMScheduler(self.config)
    
    def test_add_noise(self):
        """Test adding noise to clean samples."""
        batch_size = 4
        x = torch.randn(batch_size, 4, 32, 32)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 1000, (batch_size,))
        
        noisy_x = self.scheduler.add_noise(x, noise, timesteps)
        
        self.assertEqual(noisy_x.shape, x.shape)
        # At t=0, should be close to original
        # At t=999, should be mostly noise
    
    def test_step_deterministic(self):
        """Test DDIM deterministic sampling."""
        self.scheduler.config.eta = 0.0  # DDIM
        
        x_t = torch.randn(1, 4, 32, 32)
        noise_pred = torch.randn_like(x_t)
        timestep = 500
        
        x_t_prev, x_0_pred = self.scheduler.step(noise_pred, timestep, x_t)
        
        self.assertEqual(x_t_prev.shape, x_t.shape)
        self.assertEqual(x_0_pred.shape, x_t.shape)
    
    def test_step_stochastic(self):
        """Test DDPM stochastic sampling."""
        self.scheduler.config.eta = 1.0  # DDPM
        
        x_t = torch.randn(1, 4, 32, 32)
        noise_pred = torch.randn_like(x_t)
        timestep = 500
        
        x_t_prev, x_0_pred = self.scheduler.step(noise_pred, timestep, x_t)
        
        self.assertEqual(x_t_prev.shape, x_t.shape)
        self.assertEqual(x_0_pred.shape, x_t.shape)
    
    def test_cosine_schedule(self):
        """Test cosine noise schedule."""
        config = SchedulerConfig(
            num_train_timesteps=1000,
            schedule_type=NoiseScheduleType.COSINE,
        )
        scheduler = DDPMScheduler(config)
        
        # Alphas should be monotonically decreasing
        alphas = scheduler.alphas_cumprod.numpy()
        self.assertTrue((alphas[:-1] >= alphas[1:]).all())


class TestLatentDiffusion(unittest.TestCase):
    """Test Latent Diffusion Model."""
    
    def setUp(self):
        # Small configs for testing
        self.config = LatentDiffusionConfig()
        self.config.unet_config = UNetConfig(
            in_channels=4,
            out_channels=4,
            model_channels=64,
            channel_multipliers=(1, 2),
            num_res_blocks=1,
            attention_resolutions=(1,),
            num_heads=4,
            context_dim=128,
        )
        self.config.text_encoder_config.d_model = 128
        self.config.text_encoder_config.n_heads = 4
        self.config.text_encoder_config.n_layers = 2
        # VAE with channel_multipliers=(1, 2, 4, 8) downsamples by 2^3 = 8
        self.config.vae_config.base_channels = 32
        self.config.vae_config.channel_multipliers = (1, 2, 4, 8)
        
        self.model = LatentDiffusionModel(self.config)
        self.model.eval()
    
    def test_encode_text(self):
        """Test text encoding."""
        batch_size = 2
        seq_len = 16
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        embeddings = self.model.encode_text(input_ids)
        
        self.assertEqual(embeddings.shape, (batch_size, seq_len, 128))
    
    def test_encode_decode_image(self):
        """Test image encoding and decoding."""
        batch_size = 1
        images = torch.randn(batch_size, 3, 256, 256)
        
        # Encode (VAE with channel_multipliers=(1, 2, 4, 8) downsamples by 8)
        latents = self.model.encode_image(images)
        # With 3 downsamples: 256 / 8 = 32
        self.assertEqual(latents.shape, (batch_size, 4, 32, 32))
        
        # Decode
        reconstructed = self.model.decode_latent(latents)
        self.assertEqual(reconstructed.shape, images.shape)
    
    def test_forward_training(self):
        """Test forward pass for training."""
        batch_size = 2
        images = torch.randn(batch_size, 3, 256, 256)
        input_ids = torch.randint(0, 1000, (batch_size, 16))
        
        outputs = self.model(images, input_ids)
        
        self.assertIn('loss', outputs)
        self.assertIsInstance(outputs['loss'].item(), float)
    
    def test_generate_unconditional(self):
        """Test unconditional generation."""
        batch_size = 1
        input_ids = torch.zeros((batch_size, 16), dtype=torch.long)
        
        with torch.no_grad():
            images = self.model.generate(
                input_ids,
                height=256,
                width=256,
                num_inference_steps=10,  # Fast for testing
                guidance_scale=1.0,  # No guidance
            )
        
        self.assertEqual(images.shape, (batch_size, 3, 256, 256))
    
    def test_generate_conditional(self):
        """Test conditional generation with classifier-free guidance."""
        batch_size = 1
        input_ids = torch.randint(0, 1000, (batch_size, 16))
        
        with torch.no_grad():
            images = self.model.generate(
                input_ids,
                height=256,
                width=256,
                num_inference_steps=10,
                guidance_scale=7.5,
            )
        
        self.assertEqual(images.shape, (batch_size, 3, 256, 256))
    
    def test_num_params(self):
        """Test parameter counting."""
        num_params = self.model.get_num_params()
        self.assertGreater(num_params, 0)


if __name__ == '__main__':
    unittest.main()
