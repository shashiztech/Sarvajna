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
3. **Text-to-Image**: Latent diffusion-based image generation
4. **Text-to-Video**: Temporal video generation
5. **Image-to-Video**: Still image animation
6. **Text-to-Music**: Neural codec-based music generation

## Project Structure

```
sarvanjna/
├── data/                    # Data layer
├── preprocessing/           # Normalization and tokenization
├── models/                  # Core model architectures
│   ├── text/               # Text processor and T2T
│   ├── vision/             # Image and video models
│   └── audio/              # Music models
├── training/               # Training infrastructure
├── evaluation/             # Evaluation metrics
├── serving/                # API and inference
└── tests/                  # Unit and integration tests
```

## Development Phases

Following the recommended build sequence from the requirements:

- [x] Phase 1: Platform Foundation ✅
- [x] Phase 2: Text Core ✅
- [ ] Phase 3: Vision-Language Alignment
- [ ] Phase 4: Text-to-Image
- [ ] Phase 5: Image-to-Video
- [ ] Phase 6: Text-to-Video
- [ ] Phase 7: Text-to-Music

### ✅ Completed (Phases 1-2)

**Phase 1: Platform Foundation**
- ✅ Project structure and configuration system
- ✅ Data layer with versioning and lineage tracking
- ✅ Model registry for checkpoint management
- ✅ Distributed training infrastructure (DDP, FSDP)
- ✅ Evaluation metrics framework

**Phase 2: Text Core**
- ✅ SentencePiece tokenizer (BPE and Unigram)
- ✅ Text processor with normalization and filtering
- ✅ Complete Transformer implementation (multi-head attention, encoder, decoder)
- ✅ Text-to-Text model (T5-style encoder-decoder)
- ✅ PyTorch Lightning training module
- ✅ Text generation with sampling strategies
- ✅ Training and inference scripts
- ✅ Unit tests

### 🚧 Next Steps (Phases 3-7)

The foundation is ready for multimodal extensions. Next phases will add:
- Vision-language alignment (CLIP)
- Latent diffusion for image generation
- Temporal video models
- Music generation with neural codecs

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/
```

## License

MIT
