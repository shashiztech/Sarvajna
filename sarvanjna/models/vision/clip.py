"""
CLIP (Contrastive Language-Image Pre-training) implementation.

Based on "Learning Transferable Visual Models From Natural Language Supervision"
(Radford et al., 2021) - https://arxiv.org/abs/2103.00020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from ..text.transformer import TransformerConfig, TransformerEncoder
from .vision_transformer import VisionTransformer, ViTConfig


@dataclass
class CLIPConfig:
    """CLIP configuration."""
    
    # Vision encoder
    vision_config: ViTConfig = None
    
    # Text encoder
    text_config: TransformerConfig = None
    
    # Embedding dimension (projection space)
    embed_dim: int = 512
    
    # Temperature for contrastive loss
    temperature: float = 0.07
    
    # Whether to learn temperature
    learnable_temperature: bool = True
    
    def __post_init__(self):
        # Create default configs if not provided
        if self.vision_config is None:
            self.vision_config = ViTConfig(
                image_size=224,
                patch_size=16,
                d_model=768,
                n_heads=12,
                n_layers=12,
                num_classes=None,  # No classification head
            )
        
        if self.text_config is None:
            self.text_config = TransformerConfig(
                vocab_size=49408,  # Standard CLIP vocabulary
                d_model=512,
                n_heads=8,
                n_layers=12,
                max_seq_length=77,  # Standard CLIP text length
            )


class CLIP(nn.Module):
    """
    CLIP model for vision-language alignment.
    
    Learns joint embeddings of images and text through contrastive learning.
    """
    
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config
        
        # Vision encoder
        self.vision_encoder = VisionTransformer(config.vision_config)
        
        # Text encoder with embeddings
        self.text_embeddings = nn.Embedding(
            config.text_config.vocab_size,
            config.text_config.d_model,
        )
        self.text_encoder = TransformerEncoder(config.text_config)
        
        # Projection heads
        self.vision_projection = nn.Linear(
            config.vision_config.d_model,
            config.embed_dim,
            bias=False,
        )
        self.text_projection = nn.Linear(
            config.text_config.d_model,
            config.embed_dim,
            bias=False,
        )
        
        # Learnable temperature parameter
        if config.learnable_temperature:
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / config.temperature)))
        else:
            self.register_buffer('logit_scale', torch.log(torch.tensor(1.0 / config.temperature)))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection layers and embeddings."""
        nn.init.normal_(self.vision_projection.weight, std=0.02)
        nn.init.normal_(self.text_projection.weight, std=0.02)
        nn.init.normal_(self.text_embeddings.weight, std=0.02)
    
    def encode_image(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Encode images to embedding space.
        
        Args:
            images: (batch_size, channels, height, width)
            normalize: Whether to L2-normalize embeddings
        
        Returns:
            image_features: (batch_size, embed_dim)
        """
        # Get vision features
        vision_features = self.vision_encoder(images)  # (B, d_model)
        
        # Project to embedding space
        image_features = self.vision_projection(vision_features)  # (B, embed_dim)
        
        # Normalize
        if normalize:
            image_features = F.normalize(image_features, dim=-1)
        
        return image_features
    
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode text to embedding space.
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            normalize: Whether to L2-normalize embeddings
        
        Returns:
            text_features: (batch_size, embed_dim)
        """
        # Embed tokens
        x = self.text_embeddings(input_ids)  # (B, seq_len, d_model)
        
        # Get text features from encoder
        text_features = self.text_encoder(
            x,
            mask=attention_mask,  # TransformerEncoder uses 'mask' not 'attention_mask'
        )  # (B, seq_len, d_model)
        
        # Take the [EOS] token representation (last token)
        # In CLIP, we use the embedding at the EOS token position
        if attention_mask is not None:
            # Find last non-pad token
            seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
            text_features = text_features[torch.arange(text_features.shape[0]), seq_lengths]
        else:
            # Take last token
            text_features = text_features[:, -1, :]
        
        # Project to embedding space
        text_features = self.text_projection(text_features)  # (B, embed_dim)
        
        # Normalize
        if normalize:
            text_features = F.normalize(text_features, dim=-1)
        
        return text_features
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CLIP.
        
        Args:
            images: (batch_size, channels, height, width)
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            return_loss: Whether to compute and return contrastive loss
        
        Returns:
            Dictionary with:
                - image_features: normalized image embeddings
                - text_features: normalized text embeddings
                - logit_scale: temperature scaling parameter
                - logits_per_image: similarity scores (image -> text)
                - logits_per_text: similarity scores (text -> image)
                - loss: (optional) contrastive loss
        """
        # Encode images and text
        image_features = self.encode_image(images, normalize=True)
        text_features = self.encode_text(input_ids, attention_mask, normalize=True)
        
        # Compute similarity scores
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        output = {
            'image_features': image_features,
            'text_features': text_features,
            'logit_scale': logit_scale,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
        }
        
        if return_loss:
            loss = self.contrastive_loss(logits_per_image, logits_per_text)
            output['loss'] = loss
        
        return output
    
    def contrastive_loss(
        self,
        logits_per_image: torch.Tensor,
        logits_per_text: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute symmetric contrastive loss (InfoNCE).
        
        Args:
            logits_per_image: (batch_size, batch_size) similarity scores
            logits_per_text: (batch_size, batch_size) similarity scores
        
        Returns:
            loss: scalar loss value
        """
        batch_size = logits_per_image.shape[0]
        
        # Ground truth: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=logits_per_image.device)
        
        # Symmetric loss: image->text + text->image
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss
    
    def get_similarity(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between images and text.
        
        Args:
            images: (batch_size, channels, height, width)
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            similarity: (batch_size, batch_size) similarity scores
        """
        with torch.no_grad():
            image_features = self.encode_image(images, normalize=True)
            text_features = self.encode_text(input_ids, attention_mask, normalize=True)
            
            logit_scale = self.logit_scale.exp()
            similarity = logit_scale * image_features @ text_features.t()
        
        return similarity
    
    def zero_shot_classifier(
        self,
        class_texts: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Create zero-shot classifier weights from class descriptions.
        
        Args:
            class_texts: (num_classes, seq_len) text descriptions of classes
            attention_mask: (num_classes, seq_len)
        
        Returns:
            classifier_weights: (embed_dim, num_classes) normalized class embeddings
        """
        with torch.no_grad():
            class_embeddings = self.encode_text(class_texts, attention_mask, normalize=True)
            classifier_weights = class_embeddings.t()
        
        return classifier_weights
    
    def predict(
        self,
        images: torch.Tensor,
        classifier_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict classes using zero-shot classifier.
        
        Args:
            images: (batch_size, channels, height, width)
            classifier_weights: (embed_dim, num_classes) from zero_shot_classifier
        
        Returns:
            logits: (batch_size, num_classes) class scores
        """
        with torch.no_grad():
            image_features = self.encode_image(images, normalize=True)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ classifier_weights
        
        return logits
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


# Predefined CLIP configurations
def clip_vit_base() -> CLIP:
    """CLIP with ViT-Base vision encoder."""
    vision_config = ViTConfig(
        image_size=224,
        patch_size=16,
        d_model=768,
        n_heads=12,
        n_layers=12,
        num_classes=None,
    )
    
    text_config = TransformerConfig(
        vocab_size=49408,
        d_model=512,
        n_heads=8,
        n_layers=12,
        max_seq_length=77,
    )
    
    config = CLIPConfig(
        vision_config=vision_config,
        text_config=text_config,
        embed_dim=512,
    )
    
    return CLIP(config)


def clip_vit_large() -> CLIP:
    """CLIP with ViT-Large vision encoder."""
    vision_config = ViTConfig(
        image_size=224,
        patch_size=14,
        d_model=1024,
        n_heads=16,
        n_layers=24,
        num_classes=None,
    )
    
    text_config = TransformerConfig(
        vocab_size=49408,
        d_model=768,
        n_heads=12,
        n_layers=12,
        max_seq_length=77,
    )
    
    config = CLIPConfig(
        vision_config=vision_config,
        text_config=text_config,
        embed_dim=768,
    )
    
    return CLIP(config)
