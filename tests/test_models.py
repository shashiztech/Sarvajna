"""
Unit tests for Transformer model.
"""

import pytest
import torch

from sarvanjna.models.text.transformer import (
    TransformerConfig,
    MultiHeadAttention,
    TransformerEncoder,
    TransformerDecoder,
)
from sarvanjna.models.text.text_to_text import TextToTextModel


class TestTransformer:
    """Test Transformer components."""
    
    def test_config(self):
        config = TransformerConfig(
            vocab_size=1000,
            d_model=512,
            n_heads=8,
            n_layers=6,
        )
        
        assert config.vocab_size == 1000
        assert config.d_model == 512
        assert config.n_heads == 8
    
    def test_multihead_attention(self):
        config = TransformerConfig(d_model=512, n_heads=8)
        attn = MultiHeadAttention(config)
        
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, config.d_model)
        
        # Forward pass
        output = attn(x, x, x)
        
        assert output.shape == (batch_size, seq_len, config.d_model)
    
    def test_transformer_encoder(self):
        config = TransformerConfig(d_model=256, n_heads=4, n_layers=2)
        encoder = TransformerEncoder(config)
        
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, config.d_model)
        
        # Forward pass
        output = encoder(x)
        
        assert output.shape == (batch_size, seq_len, config.d_model)
    
    def test_transformer_decoder(self):
        config = TransformerConfig(d_model=256, n_heads=4, n_layers=2)
        decoder = TransformerDecoder(config)
        
        batch_size = 2
        src_len = 10
        tgt_len = 8
        
        x = torch.randn(batch_size, tgt_len, config.d_model)
        encoder_output = torch.randn(batch_size, src_len, config.d_model)
        
        # Forward pass
        output = decoder(x, encoder_output)
        
        assert output.shape == (batch_size, tgt_len, config.d_model)


class TestTextToTextModel:
    """Test Text-to-Text model."""
    
    def test_model_initialization(self):
        config = TransformerConfig(
            vocab_size=1000,
            d_model=256,
            n_heads=4,
            n_layers=2,
        )
        model = TextToTextModel(config)
        
        assert model.config.vocab_size == 1000
        assert model.get_num_params() > 0
    
    def test_forward_pass(self):
        config = TransformerConfig(
            vocab_size=1000,
            d_model=256,
            n_heads=4,
            n_layers=2,
        )
        model = TextToTextModel(config)
        
        batch_size = 2
        src_len = 10
        tgt_len = 8
        
        input_ids = torch.randint(0, 1000, (batch_size, src_len))
        decoder_input_ids = torch.randint(0, 1000, (batch_size, tgt_len))
        labels = torch.randint(0, 1000, (batch_size, tgt_len))
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
        
        assert "logits" in outputs
        assert "loss" in outputs
        assert outputs["logits"].shape == (batch_size, tgt_len, 1000)
        assert outputs["loss"].item() > 0
    
    def test_generate(self):
        config = TransformerConfig(
            vocab_size=1000,
            d_model=256,
            n_heads=4,
            n_layers=2,
            max_seq_length=50,
        )
        model = TextToTextModel(config)
        model.eval()
        
        batch_size = 1
        src_len = 5
        
        input_ids = torch.randint(0, 1000, (batch_size, src_len))
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=10,
            )
        
        assert output_ids.shape[0] == batch_size
        # max_length is target, actual length may vary but should be reasonable
        assert output_ids.shape[1] <= 12  # Allow some flexibility


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
