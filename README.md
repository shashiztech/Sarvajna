# Sarvanjna - Multimodal AI Platform

A production-grade multimodal AI platform built from scratch, supporting text, image, video, and music generation.

## Architecture

The platform is built as a shared foundation with modality-specific heads:

```
Data Layer → Preprocessing → Representation Learning → Generation → Evaluation → Serving
```

## Capabilities

1. **Text Processor**: Tokenization, embeddings, filtering
2. **Text-to-Text**: Summarization, Q&A, chat, translation
3. **Vision-Language Alignment**: CLIP, ViT, VAE
4. **Text-to-Image**: Latent diffusion-based image generation
5. **Image-to-Video**: Still image animation with SVD-style model
6. **Text-to-Video**: Temporal video generation from text
7. **Text-to-Music**: EnCodec + Transformer-based music generation

## Project Structure

```
sarvanjna/
├── data/                    # Data layer with versioning
├── preprocessing/           # Normalization and tokenization
├── models/                  # Core model architectures
│   ├── text/               # Text processor and Transformer
│   ├── vision/             # ViT, VAE, CLIP, U-Net, Video models
│   └── audio/              # EnCodec, MusicGen
├── training/               # PyTorch Lightning trainers
├── evaluation/             # Evaluation metrics
└── tests/                  # Unit and integration tests
```

## Development Phases

Following the recommended build sequence from the requirements:

- [x] Phase 1: Platform Foundation ✅
- [x] Phase 2: Text Core ✅
- [x] Phase 3: Vision-Language Alignment ✅
- [x] Phase 4: Text-to-Image ✅
- [x] Phase 5: Image-to-Video ✅
- [x] Phase 6: Text-to-Video ✅
- [x] Phase 7: Text-to-Music ✅

### ✅ All Phases Completed!

**Phase 1: Platform Foundation**
- ✅ Configuration system with OmegaConf
- ✅ Data layer with versioning and lineage tracking
- ✅ Model registry for checkpoint management
- ✅ Distributed training infrastructure (DDP, FSDP, mixed precision)
- ✅ Evaluation metrics framework

**Phase 2: Text Core**
- ✅ SentencePiece tokenizer (BPE and Unigram)
- ✅ Text processor with normalization and PII filtering
- ✅ Complete Transformer (multi-head attention, encoder, decoder, positional encoding)
- ✅ Text-to-Text model (T5-style encoder-decoder)
- ✅ PyTorch Lightning training module
- ✅ Text generation with multiple sampling strategies
- ✅ Training and inference scripts

**Phase 3: Vision-Language Alignment**
- ✅ Vision Transformer (ViT) - 4 variants: tiny (~5M), small (~22M), base (~86M), large (~304M params)
- ✅ Image VAE for latent compression
- ✅ CLIP contrastive learning model
- ✅ Training modules for CLIP and VAE
- ✅ Image-text paired dataset

**Phase 4: Text-to-Image**
- ✅ U-Net architecture for diffusion in latent space
- ✅ DDPM/DDIM noise schedulers
- ✅ Latent Diffusion Model with cross-attention text conditioning
- ✅ Classifier-free guidance for generation
- ✅ Training and inference scripts
- ✅ Unit tests (13 tests passing)

**Phase 5: Image-to-Video**
- ✅ Temporal layers (3D convolutions, temporal attention)
- ✅ Video VAE for video compression
- ✅ Image-to-Video model with SVD-style conditioning
- ✅ Frame interpolation and motion control

**Phase 6: Text-to-Video**
- ✅ Text-to-Video U-Net with spatial-temporal attention
- ✅ Cross-attention for text conditioning
- ✅ Video generation from text prompts
- ✅ Temporal coherence mechanisms

**Phase 7: Text-to-Music**
- ✅ EnCodec neural audio codec
  - Residual Vector Quantization (RVQ)
  - High-fidelity audio compression
- ✅ MusicGen Transformer LM
  - Autoregressive generation over audio tokens
  - Text conditioning with CFG
  - Multi-codebook delay pattern

## Key Features

### Text Generation
- Multiple tokenization strategies
- Encoder-decoder and decoder-only architectures
- Greedy, beam search, top-k, nucleus sampling

### Image Generation
- Latent diffusion for efficient training
- Classifier-free guidance
- DDIM fast sampling
- VAE for 8x spatial compression

### Video Generation
- Temporal coherence across frames
- Text and image conditioning
- Progressive generation
- 3D U-Net architecture

### Music Generation
- Neural audio codec at 24kHz
- Multi-codebook quantization
- Text-to-music with controllability
- Long-form generation (10+ seconds)

## Technical Stack

- **Framework**: PyTorch 2.0+, PyTorch Lightning 2.0+
- **Tokenization**: SentencePiece
- **Training**: DDP, FSDP, bf16-mixed precision
- **Config**: OmegaConf, Hydra
- **Logging**: TensorBoard, WandB

## Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/  # 28+ tests passing
```

## Usage Examples

### Text Generation
```bash
python examples/inference_text.py \
  --checkpoint checkpoints/text_model.ckpt \
  --prompt "Translate to French: Hello world"
```

### Image Generation
```bash
python examples/inference_latent_diffusion.py \
  --checkpoint checkpoints/ldm.ckpt \
  --prompt "A beautiful sunset over mountains" \
  --output output.png \
  --num-steps 50 \
  --guidance-scale 7.5
```

## Model Statistics

| Model | Parameters | Description |
|-------|-----------|-------------|
| Text-to-Text Base | ~60M | Encoder-decoder Transformer |
| ViT-Large | ~304M | Vision encoder |
| VAE | ~83M | Image compression |
| CLIP | ~150M | Vision-language alignment |
| U-Net (LDM) | ~860M | Latent diffusion |
| Video U-Net | ~1.2B | Text-to-video generation |
| MusicGen | ~300M | Audio generation |

## References

This implementation is based on:
- "Attention is All You Need" (Vaswani et al., 2017)
- "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
- "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)
- "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022)
- "Stable Video Diffusion" (Blattmann et al., 2023)
- "High Fidelity Neural Audio Compression" (Défossez et al., 2022)
- "Simple and Controllable Music Generation" (Copet et al., 2023)

## License

MIT
