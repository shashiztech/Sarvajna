"""
Transformer architecture implementation.

Based on "Attention is All You Need" (Vaswani et al., 2017)
https://arxiv.org/abs/1706.03762
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TransformerConfig:
    """Configuration for Transformer models."""
    vocab_size: int = 32000
    d_model: int = 768  # Model dimension
    n_heads: int = 12  # Number of attention heads
    n_layers: int = 12  # Number of layers
    d_ff: int = 3072  # Feedforward dimension (typically 4 * d_model)
    dropout: float = 0.1
    max_seq_length: int = 512
    layer_norm_eps: float = 1e-6
    activation: str = "gelu"  # or "relu"
    pad_token_id: int = 0
    

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    From "Attention is All You Need"
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        
        # Linear projections
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            query: (batch, seq_len_q, d_model)
            key: (batch, seq_len_k, d_model)
            value: (batch, seq_len_v, d_model)
            mask: (batch, seq_len_q, seq_len_k) or (batch, 1, seq_len_q, seq_len_k)
            
        Returns:
            Output: (batch, seq_len_q, d_model)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape to (batch, n_heads, seq_len, d_k)
        Q = self.q_proj(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        # scores: (batch, n_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask to match attention scores shape
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, seq_len_q, seq_len_k)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # output: (batch, n_heads, seq_len_q, d_k)
        output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and project
        # (batch, seq_len_q, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(output)
        
        return output


class FeedForward(nn.Module):
    """Position-wise feedforward network."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        self.activation = F.gelu if config.activation == "gelu" else F.relu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            Output: (batch, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with pre-norm (like T5).
        
        Args:
            x: (batch, seq_len, d_model)
            mask: Attention mask
            
        Returns:
            Output: (batch, seq_len, d_model)
        """
        # Self-attention with residual
        attn_out = self.attention(x, x, x, mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        
        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        
        return x


class TransformerDecoderLayer(nn.Module):
    """Single Transformer decoder layer."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.cross_attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm3 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Decoder input (batch, seq_len_dec, d_model)
            encoder_output: Encoder output (batch, seq_len_enc, d_model)
            self_attn_mask: Causal mask for self-attention
            cross_attn_mask: Mask for cross-attention
            
        Returns:
            Output: (batch, seq_len_dec, d_model)
        """
        # Self-attention with residual
        self_attn_out = self.self_attention(x, x, x, self_attn_mask)
        x = x + self.dropout1(self_attn_out)
        x = self.norm1(x)
        
        # Cross-attention with residual
        cross_attn_out = self.cross_attention(x, encoder_output, encoder_output, cross_attn_mask)
        x = x + self.dropout2(cross_attn_out)
        x = self.norm2(x)
        
        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = x + self.dropout3(ff_out)
        x = self.norm3(x)
        
        return x


class TransformerEncoder(nn.Module):
    """Stack of Transformer encoder layers."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config)
            for _ in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through all encoder layers.
        
        Args:
            x: (batch, seq_len, d_model)
            mask: Attention mask
            
        Returns:
            Encoder output: (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        return x


class TransformerDecoder(nn.Module):
    """Stack of Transformer decoder layers."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config)
            for _ in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through all decoder layers.
        
        Args:
            x: Decoder input (batch, seq_len_dec, d_model)
            encoder_output: Encoder output (batch, seq_len_enc, d_model)
            self_attn_mask: Causal mask for self-attention
            cross_attn_mask: Mask for cross-attention
            
        Returns:
            Decoder output: (batch, seq_len_dec, d_model)
        """
        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_mask, cross_attn_mask)
        
        x = self.norm(x)
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)
