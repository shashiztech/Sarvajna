"""
Unit tests for vision models.
"""

import pytest
import torch

from sarvanjna.models.vision.vision_transformer import (
    VisionTransformer,
    ViTConfig,
    vit_tiny,
    vit_base,
)
from sarvanjna.models.vision.image_autoencoder import (
    ImageAutoencoder,
    VAEConfig,
)
from sarvanjna.models.vision.clip import (
    CLIP,
    CLIPConfig,
    clip_vit_base,
)


class TestVisionTransformer:
    """Test Vision Transformer."""
    
    def test_config(self):
        config = ViTConfig(
            image_size=224,
            patch_size=16,
            d_model=768,
            n_heads=12,
            n_layers=12,
        )
        
        assert config.num_patches == (224 // 16) ** 2
        assert config.num_patches == 196
    
    def test_forward_pass(self):
        config = ViTConfig(
            image_size=224,
            patch_size=16,
            d_model=256,
            n_heads=4,
            n_layers=2,
            num_classes=None,
        )
        model = VisionTransformer(config)
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)
        
        # Forward pass
        output = model(x)
        
        assert output.shape == (batch_size, config.d_model)
    
    def test_with_classification_head(self):
        config = ViTConfig(
            image_size=224,
            patch_size=16,
            d_model=256,
            n_heads=4,
            n_layers=2,
            num_classes=1000,
        )
        model = VisionTransformer(config)
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)
        
        # Forward pass
        output = model(x)
        
        assert output.shape == (batch_size, 1000)
    
    def test_return_features(self):
        config = ViTConfig(
            image_size=224,
            patch_size=16,
            d_model=256,
            n_heads=4,
            n_layers=2,
        )
        model = VisionTransformer(config)
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)
        
        # Get all features
        features = model(x, return_features=True)
        
        # Should include CLS token + patches
        assert features.shape == (batch_size, config.num_patches + 1, config.d_model)
    
    def test_vit_variants(self):
        # Test predefined configurations
        model_tiny = vit_tiny()
        model_base = vit_base()
        
        x = torch.randn(1, 3, 224, 224)
        
        out_tiny = model_tiny(x)
        out_base = model_base(x)
        
        assert out_tiny.shape == (1, 192)  # d_model for tiny
        assert out_base.shape == (1, 768)  # d_model for base


class TestImageAutoencoder:
    """Test Image VAE."""
    
    def test_config(self):
        config = VAEConfig(
            latent_channels=4,
            base_channels=128,
            resolution=256,
        )
        
        assert config.latent_channels == 4
        assert config.resolution == 256
    
    def test_forward_pass(self):
        config = VAEConfig(
            latent_channels=4,
            base_channels=64,  # Smaller for testing
            channel_multipliers=(1, 2),  # Fewer layers
            resolution=128,  # Smaller resolution
        )
        model = ImageAutoencoder(config)
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 128, 128)
        
        # Forward pass
        outputs = model(x, return_latent=True)
        
        assert 'reconstruction' in outputs
        assert 'loss' in outputs
        assert 'kl_loss' in outputs
        assert 'recon_loss' in outputs
        assert outputs['reconstruction'].shape == x.shape
    
    def test_encode_decode(self):
        config = VAEConfig(
            latent_channels=4,
            base_channels=64,
            channel_multipliers=(1, 2),
            resolution=128,
        )
        model = ImageAutoencoder(config)
        model.eval()
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 128, 128)
        
        # Encode
        with torch.no_grad():
            z = model.encode(x, sample=False)
            
            # Check latent size (downsampled once since we have 2 stages, one downsample between them)
            num_downsamples = len(config.channel_multipliers) - 1
            expected_h = expected_w = 128 // (2 ** num_downsamples)
            assert z.shape == (batch_size, config.latent_channels, expected_h, expected_w)
            
            # Decode
            recon = model.decode(z)
            assert recon.shape == x.shape


class TestCLIP:
    """Test CLIP model."""
    
    def test_config(self):
        config = CLIPConfig()
        
        assert config.vision_config is not None
        assert config.text_config is not None
        assert config.embed_dim == 512
    
    def test_forward_pass(self):
        from sarvanjna.models.vision.vision_transformer import ViTConfig
        from sarvanjna.models.text.transformer import TransformerConfig
        
        vision_config = ViTConfig(
            image_size=224,
            patch_size=16,
            d_model=256,
            n_heads=4,
            n_layers=2,
            num_classes=None,
        )
        
        text_config = TransformerConfig(
            vocab_size=1000,
            d_model=256,
            n_heads=4,
            n_layers=2,
            max_seq_length=77,
        )
        
        clip_config = CLIPConfig(
            vision_config=vision_config,
            text_config=text_config,
            embed_dim=256,
        )
        
        model = CLIP(clip_config)
        
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (batch_size, 77))
        
        # Forward pass
        outputs = model(images, input_ids, return_loss=True)
        
        assert 'image_features' in outputs
        assert 'text_features' in outputs
        assert 'loss' in outputs
        assert outputs['image_features'].shape == (batch_size, 256)
        assert outputs['text_features'].shape == (batch_size, 256)
    
    def test_encode_image(self):
        config = CLIPConfig()
        model = CLIP(config)
        model.eval()
        
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            features = model.encode_image(images)
        
        assert features.shape == (batch_size, config.embed_dim)
        
        # Check normalization
        norms = torch.norm(features, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_encode_text(self):
        config = CLIPConfig()
        model = CLIP(config)
        model.eval()
        
        batch_size = 2
        input_ids = torch.randint(0, config.text_config.vocab_size, (batch_size, 77))
        
        with torch.no_grad():
            features = model.encode_text(input_ids)
        
        assert features.shape == (batch_size, config.embed_dim)
        
        # Check normalization
        norms = torch.norm(features, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_contrastive_loss(self):
        config = CLIPConfig()
        model = CLIP(config)
        
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(0, config.text_config.vocab_size, (batch_size, 77))
        
        # Forward pass
        outputs = model(images, input_ids, return_loss=True)
        loss = outputs['loss']
        
        # Loss should be positive
        assert loss.item() > 0
        
        # Loss should be a scalar
        assert loss.shape == ()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
