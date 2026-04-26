"""
MusicGen: Text-to-Music generation model.

Based on "Simple and Controllable Music Generation" (Copet et al., 2023).
Uses autoregressive Transformer over EnCodec tokens.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, List
from tqdm import tqdm

from .audio_codec import EnCodec, EnCodecConfig
from ..text.transformer import TransformerDecoder, TransformerConfig


@dataclass
class MusicGenConfig:
    """Configuration for MusicGen."""
    
    # Audio codec
    codec_config: EnCodecConfig = None
    
    # Transformer
    transformer_config: TransformerConfig = None
    
    # Conditioning
    text_vocab_size: int = 49408
    text_max_length: int = 77
    text_embed_dim: int = 768
    
    # Audio generation
    sample_rate: int = 24000
    duration: float = 10.0  # seconds
    
    # Training
    use_cfg: bool = True  # Classifier-free guidance
    cfg_dropout: float = 0.1
    
    def __post_init__(self):
        if self.codec_config is None:
            self.codec_config = EnCodecConfig()
        
        if self.transformer_config is None:
            # Transformer over audio tokens
            self.transformer_config = TransformerConfig(
                vocab_size=self.codec_config.codebook_size,
                d_model=512,
                n_heads=8,
                n_layers=12,
                d_ff=2048,
                max_seq_length=1500,  # ~10 seconds at 24kHz
            )


class DelayPattern(nn.Module):
    """
    Delay pattern for parallel prediction of multiple codebook streams.
    
    Implements the delay pattern from MusicGen paper for efficient
    multi-stream generation.
    """
    
    def __init__(self, num_codebooks: int):
        super().__init__()
        self.num_codebooks = num_codebooks
    
    def apply_delay(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Apply delay pattern to codes.
        
        Args:
            codes: (batch, num_codebooks, time)
        
        Returns:
            delayed: (batch, num_codebooks, time + delays)
        """
        B, K, T = codes.shape
        
        # Pad each codebook stream with different delays
        delayed_codes = []
        for k in range(K):
            delay = k
            padded = F.pad(codes[:, k:k+1], (delay, 0), value=0)
            delayed_codes.append(padded)
        
        delayed = torch.cat(delayed_codes, dim=1)
        
        return delayed
    
    def remove_delay(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Remove delay pattern from codes.
        
        Args:
            codes: (batch, num_codebooks, time + delays)
        
        Returns:
            undelayed: (batch, num_codebooks, time)
        """
        B, K, T_delayed = codes.shape
        
        undelayed_codes = []
        for k in range(K):
            delay = k
            undelayed = codes[:, k:k+1, delay:]
            undelayed_codes.append(undelayed)
        
        # Find minimum length
        min_length = min(c.shape[2] for c in undelayed_codes)
        
        # Truncate all to same length
        undelayed_codes = [c[:, :, :min_length] for c in undelayed_codes]
        undelayed = torch.cat(undelayed_codes, dim=1)
        
        return undelayed


class MusicGen(nn.Module):
    """
    MusicGen: Text-to-Music generation model.
    
    Combines EnCodec with autoregressive Transformer LM.
    """
    
    def __init__(self, config: MusicGenConfig):
        super().__init__()
        self.config = config
        
        # Audio codec
        self.codec = EnCodec(config.codec_config)
        
        # Freeze codec during training
        for param in self.codec.parameters():
            param.requires_grad = False
        
        # Text encoder
        self.text_embeddings = nn.Embedding(config.text_vocab_size, config.text_embed_dim)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.text_embed_dim,
                nhead=8,
                dim_feedforward=config.text_embed_dim * 4,
                batch_first=True,
            ),
            num_layers=6,
        )
        
        # Audio token embeddings (for each codebook)
        self.audio_embeddings = nn.ModuleList([
            nn.Embedding(config.codec_config.codebook_size, config.transformer_config.d_model)
            for _ in range(config.codec_config.num_codebooks)
        ])
        
        # Cross-attention for text conditioning
        self.cross_attn_proj = nn.Linear(config.text_embed_dim, config.transformer_config.d_model)
        
        # Transformer decoder
        self.decoder = TransformerDecoder(config.transformer_config)
        
        # Output heads (one per codebook)
        self.output_heads = nn.ModuleList([
            nn.Linear(config.transformer_config.d_model, config.codec_config.codebook_size)
            for _ in range(config.codec_config.num_codebooks)
        ])
        
        # Delay pattern
        self.delay_pattern = DelayPattern(config.codec_config.num_codebooks)
        
        # CFG
        self.use_cfg = config.use_cfg
        self.cfg_dropout = config.cfg_dropout
    
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode text prompt.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
        
        Returns:
            text_embeddings: (batch, seq_len, d_model)
        """
        # Embed
        x = self.text_embeddings(input_ids)
        
        # Encode
        if attention_mask is not None:
            # Convert to float mask for transformer
            key_padding_mask = (attention_mask == 0)
            text_embeddings = self.text_encoder(x, src_key_padding_mask=key_padding_mask)
        else:
            text_embeddings = self.text_encoder(x)
        
        return text_embeddings
    
    def forward(
        self,
        audio: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            audio: (batch, 1, time) waveform
            input_ids: (batch, seq_len) text tokens
            attention_mask: (batch, seq_len)
        
        Returns:
            Dictionary with loss
        """
        batch_size = audio.shape[0]
        
        # Encode audio to codes
        with torch.no_grad():
            codes, _ = self.codec.encode(audio)  # (batch, num_codebooks, time_compressed)
        
        # CFG: randomly drop text conditioning
        if self.use_cfg and self.training:
            drop_mask = torch.rand(batch_size, device=audio.device) < self.cfg_dropout
            input_ids = input_ids * (~drop_mask).unsqueeze(-1)
        
        # Encode text
        text_embeddings = self.encode_text(input_ids, attention_mask)
        text_embeddings = self.cross_attn_proj(text_embeddings)
        
        # Apply delay pattern
        codes_delayed = self.delay_pattern.apply_delay(codes)
        
        # Embed audio tokens
        B, K, T = codes_delayed.shape
        audio_emb = torch.zeros(B, T, self.config.transformer_config.d_model, device=audio.device)
        
        for k in range(K):
            audio_emb = audio_emb + self.audio_embeddings[k](codes_delayed[:, k])
        
        # Decode with cross-attention to text
        # Note: TransformerDecoder expects (seq_len, batch, d_model) format
        audio_emb = audio_emb.transpose(0, 1)
        text_embeddings = text_embeddings.transpose(0, 1)
        
        # Create causal mask
        seq_len = audio_emb.shape[0]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=audio.device), diagonal=1).bool()
        
        # Decode (simplified - full implementation would use cross-attention)
        hidden = self.decoder(audio_emb, memory=text_embeddings, mask=causal_mask)
        hidden = hidden.transpose(0, 1)  # (batch, seq_len, d_model)
        
        # Predict next tokens for each codebook
        logits_list = []
        for k, head in enumerate(self.output_heads):
            logits = head(hidden)
            logits_list.append(logits)
        
        logits = torch.stack(logits_list, dim=1)  # (batch, num_codebooks, seq_len, vocab_size)
        
        # Compute loss (next token prediction for each codebook)
        # Shift targets
        targets = codes_delayed[:, :, 1:]  # (batch, num_codebooks, seq_len-1)
        logits_shifted = logits[:, :, :-1]  # (batch, num_codebooks, seq_len-1, vocab_size)
        
        # Cross-entropy loss
        loss = 0
        for k in range(K):
            loss_k = F.cross_entropy(
                logits_shifted[:, k].reshape(-1, self.config.codec_config.codebook_size),
                targets[:, k].reshape(-1),
            )
            loss = loss + loss_k
        
        loss = loss / K
        
        return {
            'loss': loss,
            'logits': logits,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        duration: Optional[float] = None,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        cfg_scale: float = 3.0,
    ) -> torch.Tensor:
        """
        Generate music from text prompt.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            duration: duration in seconds
            temperature: sampling temperature
            top_k: top-k sampling
            top_p: nucleus sampling
            cfg_scale: classifier-free guidance scale
        
        Returns:
            audio: (batch, 1, time) waveform
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        duration = duration or self.config.duration
        
        # Calculate number of tokens needed
        codec_fps = self.config.sample_rate / (
            torch.prod(torch.tensor(self.config.codec_config.encoder_strides)).item()
        )
        num_tokens = int(duration * codec_fps)
        
        # Encode text (conditional)
        text_embeddings = self.encode_text(input_ids, attention_mask)
        text_embeddings = self.cross_attn_proj(text_embeddings)
        
        # Encode text (unconditional) for CFG
        if cfg_scale > 1.0:
            uncond_input_ids = torch.zeros_like(input_ids)
            uncond_embeddings = self.encode_text(uncond_input_ids)
            uncond_embeddings = self.cross_attn_proj(uncond_embeddings)
        
        # Initialize with start tokens
        K = self.config.codec_config.num_codebooks
        codes = torch.zeros(batch_size, K, 1, dtype=torch.long, device=device)
        
        # Autoregressive generation
        for i in tqdm(range(num_tokens), desc="Generating music"):
            # Apply delay pattern
            codes_delayed = self.delay_pattern.apply_delay(codes)
            
            # Embed
            B, K, T = codes_delayed.shape
            audio_emb = torch.zeros(B, T, self.config.transformer_config.d_model, device=device)
            
            for k in range(K):
                audio_emb = audio_emb + self.audio_embeddings[k](codes_delayed[:, k])
            
            # Decode
            audio_emb = audio_emb.transpose(0, 1)
            text_emb_cond = text_embeddings.transpose(0, 1)
            
            # Conditional
            hidden_cond = self.decoder(audio_emb, memory=text_emb_cond)
            hidden_cond = hidden_cond.transpose(0, 1)[:, -1]  # Last token
            
            # Predict next tokens
            logits_list = []
            for k, head in enumerate(self.output_heads):
                logits_k = head(hidden_cond)
                
                # CFG
                if cfg_scale > 1.0:
                    text_emb_uncond = uncond_embeddings.transpose(0, 1)
                    hidden_uncond = self.decoder(audio_emb, memory=text_emb_uncond)
                    hidden_uncond = hidden_uncond.transpose(0, 1)[:, -1]
                    logits_k_uncond = head(hidden_uncond)
                    
                    logits_k = logits_k_uncond + cfg_scale * (logits_k - logits_k_uncond)
                
                logits_list.append(logits_k)
            
            logits = torch.stack(logits_list, dim=1)  # (batch, num_codebooks, vocab_size)
            
            # Sample next tokens
            logits = logits / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < v[:, :, [-1]]] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1)
            next_tokens = next_tokens.view(batch_size, K, 1)
            
            # Append
            codes = torch.cat([codes, next_tokens], dim=2)
        
        # Remove delay pattern
        codes = self.delay_pattern.remove_delay(codes)
        
        # Decode to audio
        quantized = self.codec.quantizer.decode(codes)
        audio = self.codec.decode(quantized)
        
        return audio
    
    def get_num_params(self) -> int:
        """Get total number of parameters (excluding codec)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
