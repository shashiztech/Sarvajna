# Sarvanjna - Quick Start Guide

This guide will help you get started with the Sarvanjna multimodal AI platform.

## Installation

1. **Clone the repository**:
```bash
cd Sarvanjna
```

2. **Create virtual environment**:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install -e .
```

## Phase 1: Text Processing

### Step 1: Train a Tokenizer

First, you need to train a SentencePiece tokenizer on your text corpus.

```bash
# Prepare your text corpus (one sentence per line)
# Example: data/corpus.txt

# Train tokenizer
python examples/train_tokenizer.py \
    --input data/corpus.txt \
    --output models/tokenizer \
    --vocab_size 32000 \
    --model_type unigram
```

This creates `models/tokenizer.model` and `models/tokenizer.vocab`.

### Step 2: Prepare Training Data

Format your training data as JSON or JSONL:

```json
[
  {
    "instruction": "Summarize this text:",
    "input": "Long text to summarize...",
    "output": "Summary of the text."
  },
  {
    "instruction": "Translate to French:",
    "input": "Hello world",
    "output": "Bonjour le monde"
  }
]
```

Save as `data/train.json` and `data/val.json`.

### Step 3: Train Text-to-Text Model

```bash
python examples/train_text_model.py \
    --config configs/text_base.yaml \
    --data_path data/train.json \
    --val_data_path data/val.json \
    --tokenizer_model models/tokenizer.model \
    --batch_size 32 \
    --max_epochs 10 \
    --output_dir outputs/text-model
```

### Step 4: Run Inference

```bash
python examples/inference_text.py \
    --checkpoint checkpoints/text-model-epoch=09-val_loss=2.3456.ckpt \
    --tokenizer models/tokenizer.model \
    --temperature 0.9 \
    --do_sample
```

## Configuration

Edit `configs/text_base.yaml` to customize:

- **Model size**: `d_model`, `n_heads`, `n_layers`
- **Training**: `learning_rate`, `batch_size`, `max_steps`
- **Distributed**: `strategy` (ddp, fsdp), `devices` (number of GPUs)

### Model Sizes

| Config | Parameters | d_model | n_layers | Memory |
|--------|-----------|---------|----------|---------|
| Small  | ~60M      | 512     | 6        | ~8 GB   |
| Base   | ~220M     | 768     | 12       | ~16 GB  |
| Large  | ~770M     | 1024    | 24       | ~32 GB  |

## Project Structure

```
sarvanjna/
├── core/              # Configuration and registry
├── data/              # Dataset classes
├── preprocessing/     # Text processing and tokenization
├── models/
│   ├── text/         # Transformer models
│   ├── vision/       # Image/video models (coming soon)
│   └── audio/        # Music models (coming soon)
├── training/         # Training loops
├── evaluation/       # Metrics
└── serving/          # API (coming soon)
```

## Next Steps

After completing Phase 1 (Text Processing), you can proceed to:

- **Phase 2**: Vision-Language Alignment (CLIP-style)
- **Phase 3**: Text-to-Image Generation
- **Phase 4**: Video Generation
- **Phase 5**: Music Generation

## Distributed Training

### Multi-GPU (Single Node)

```yaml
# configs/text_base.yaml
training:
  strategy: ddp
  devices: -1  # Use all GPUs
  precision: bf16-mixed
```

### Multi-Node (Cluster)

```yaml
training:
  strategy: fsdp  # For large models
  num_nodes: 4
  devices: 8
```

Run with:
```bash
python examples/train_text_model.py \
    --config configs/text_large.yaml \
    ...
```

## Monitoring

Training metrics are logged to:
- **Console**: Real-time progress
- **WandB**: Online dashboard (if enabled)
- **TensorBoard**: Local logs

View TensorBoard:
```bash
tensorboard --logdir outputs/
```

## Tips

1. **Start small**: Use `text_base.yaml` for initial experiments
2. **Monitor GPU memory**: Reduce `batch_size` if OOM
3. **Use mixed precision**: `bf16-mixed` for faster training
4. **Checkpoint often**: Save every 5000 steps
5. **Validate regularly**: Check validation loss to avoid overfitting

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use gradient accumulation
- Enable gradient checkpointing
- Use FSDP for large models

### Slow Training
- Increase number of data workers
- Use mixed precision
- Enable pin_memory in DataLoader
- Profile with PyTorch Profiler

### Poor Quality
- Increase model size
- Train longer
- Improve data quality
- Adjust learning rate

## Resources

- Documentation: `docs/`
- Examples: `examples/`
- Tests: `tests/`
- Configs: `configs/`

## Support

For issues, questions, or contributions, see [CONTRIBUTING.md](CONTRIBUTING.md).
