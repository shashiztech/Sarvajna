"""
Text-to-Text model (T5-style encoder-decoder).

Supports:
- Summarization
- Translation
- Question answering
- Text classification as generation
- General text transformation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

from sarvanjna.models.text.transformer import (
    TransformerConfig,
    TransformerEncoder,
    TransformerDecoder,
    PositionalEncoding,
)


class TextToTextModel(nn.Module):
    """
    Text-to-Text Transformer model.
    
    Encoder-decoder architecture that treats all NLP tasks as text-to-text.
    Based on T5 (Raffel et al., 2019): https://arxiv.org/abs/1910.10683
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Shared embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_length, config.dropout)
        
        # Encoder and decoder
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        
        # Output projection (shared with embedding for weight tying)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying (share embeddings with output projection)
        self.lm_head.weight = self.embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Encoder input (batch, src_len)
            decoder_input_ids: Decoder input (batch, tgt_len)
            attention_mask: Encoder attention mask (batch, src_len)
            decoder_attention_mask: Decoder attention mask (batch, tgt_len)
            labels: Target labels for computing loss (batch, tgt_len)
            
        Returns:
            Dictionary with logits and optional loss
        """
        # Encode
        encoder_output = self.encode(input_ids, attention_mask)
        
        # If no decoder input provided (inference), return encoder output
        if decoder_input_ids is None:
            return {"encoder_output": encoder_output}
        
        # Decode
        decoder_output = self.decode(
            decoder_input_ids,
            encoder_output,
            decoder_attention_mask,
            attention_mask,
        )
        
        # Project to vocabulary
        logits = self.lm_head(decoder_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
        
        return {
            "logits": logits,
            "loss": loss,
            "encoder_output": encoder_output,
            "decoder_output": decoder_output,
        }
    
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode input text.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            
        Returns:
            Encoder output: (batch, seq_len, d_model)
        """
        # Embed and add positional encoding
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        
        # Create attention mask for encoder
        encoder_mask = None
        if attention_mask is not None:
            # Convert to (batch, 1, 1, seq_len) for broadcasting
            encoder_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Encode
        encoder_output = self.encoder(x, encoder_mask)
        
        return encoder_output
    
    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode given encoder output.
        
        Args:
            decoder_input_ids: (batch, tgt_len)
            encoder_output: (batch, src_len, d_model)
            decoder_attention_mask: (batch, tgt_len)
            encoder_attention_mask: (batch, src_len)
            
        Returns:
            Decoder output: (batch, tgt_len, d_model)
        """
        # Embed and add positional encoding
        x = self.embedding(decoder_input_ids)
        x = self.pos_encoding(x)
        
        # Create causal mask for decoder self-attention
        tgt_len = decoder_input_ids.size(1)
        causal_mask = self._generate_causal_mask(tgt_len, decoder_input_ids.device)
        
        # Combine with padding mask if provided
        if decoder_attention_mask is not None:
            # (batch, 1, tgt_len, tgt_len)
            padding_mask = decoder_attention_mask.unsqueeze(1).unsqueeze(2)
            self_attn_mask = causal_mask & padding_mask
        else:
            self_attn_mask = causal_mask
        
        # Create cross-attention mask
        cross_attn_mask = None
        if encoder_attention_mask is not None:
            # (batch, 1, 1, src_len)
            cross_attn_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Decode
        decoder_output = self.decoder(
            x,
            encoder_output,
            self_attn_mask,
            cross_attn_mask,
        )
        
        return decoder_output
    
    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal mask for autoregressive decoding.
        
        Args:
            size: Sequence length
            device: Torch device
            
        Returns:
            Causal mask: (1, 1, size, size)
        """
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        mask = ~mask  # Invert: True = attend, False = mask
        return mask.unsqueeze(0).unsqueeze(0)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = False,
        eos_token_id: int = 3,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Input prompt (batch, src_len)
            attention_mask: Attention mask
            max_length: Maximum generation length
            num_beams: Beam search width (1 = greedy)
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample or use greedy/beam
            eos_token_id: End-of-sequence token
            pad_token_id: Padding token
            
        Returns:
            Generated token IDs (batch, gen_len)
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Encode input
        encoder_output = self.encode(input_ids, attention_mask)
        
        # Initialize decoder input with BOS token (assuming BOS = 2)
        decoder_input_ids = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)
        
        # Generate tokens
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length):
            # Decode
            decoder_output = self.decode(
                decoder_input_ids,
                encoder_output,
                encoder_attention_mask=attention_mask,
            )
            
            # Get logits for last token
            logits = self.lm_head(decoder_output[:, -1, :])  # (batch, vocab_size)
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Sample or select next token
            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for i in range(batch_size):
                        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                        logits[i, indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            # Append to sequence
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            
            # Check for EOS
            finished |= (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break
        
        return decoder_input_ids
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
