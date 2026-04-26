# Continuous Learning System

Automated training pipeline for Sarvanjna AI models with legal data collection and incremental learning.

## Overview

The continuous learning system enables automated, ongoing improvement of AI models by:
- Collecting training data from legal sources (Wikipedia, Project Gutenberg, Wikimedia Commons)
- Preprocessing and quality-filtering the data
- Training models incrementally on new data
- Evaluating model performance
- Versioning and deploying improved models
- Tracking all data sources and licenses for legal compliance

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Continuous Learning Pipeline               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Data      │───▶│ Preprocessing│───▶│   Training   │  │
│  │ Collection  │    │              │    │              │  │
│  └─────────────┘    └──────────────┘    └──────────────┘  │
│         │                                        │          │
│         ▼                                        ▼          │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Legal     │    │   Quality    │    │  Evaluation  │  │
│  │  Tracking   │    │   Filtering  │    │              │  │
│  └─────────────┘    └──────────────┘    └──────────────┘  │
│                                                  │          │
│                                                  ▼          │
│                                           ┌──────────────┐  │
│                                           │  Deployment  │  │
│                                           └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Legal Data Collector (`scripts/legal_data_collector.py`)

Collects training data from legal sources:

**Wikipedia**
- License: CC-BY-SA 3.0
- Content: Articles from specified categories
- Rate limit: 1 second between requests

**Project Gutenberg**
- License: Public Domain
- Content: Books published before copyright expiration
- Formats: Plain text (.txt)

**Wikimedia Commons**
- Licenses: CC0, CC-BY-4.0, CC-BY-SA-4.0, CC-BY-SA-3.0
- Content: Images from specified categories
- Quality: High-resolution images preferred

**Features:**
- Automatic license tracking
- Attribution metadata
- Data manifest generation
- Rate limiting to respect servers

### 2. Continuous Learning Pipeline (`scripts/continuous_learning_pipeline.py`)

Orchestrates the full training lifecycle:

**Pipeline Stages:**
1. **Data Collection** - Runs legal_data_collector.py
2. **Preprocessing** - Cleans and normalizes data
3. **Training** - Incremental model training
4. **Evaluation** - Computes metrics (perplexity, BLEU, FID, etc.)
5. **Versioning** - Tracks model versions and performance
6. **Deployment** - Deploys improved models

**Features:**
- Incremental learning (continues from previous checkpoint)
- Model versioning with metrics tracking
- Automatic deployment of improved models
- State persistence across runs
- Scheduled execution (hourly, daily, weekly)

## Configuration

Configuration file: `configs/continuous_learning.yaml`

### Data Collection Settings

```yaml
data_collection:
  wikipedia:
    enabled: true
    categories: ["Machine learning", "Artificial intelligence"]
    max_articles: 1000
  
  gutenberg:
    enabled: true
    max_books: 100
  
  wikimedia:
    enabled: true
    max_images: 10000
    allowed_licenses: ["CC0", "CC-BY-4.0", "CC-BY-SA-4.0"]
```

### Training Settings

```yaml
training:
  text:
    batch_size: 32
    max_epochs: 10
    learning_rate: 1.0e-4
    incremental: true
```

### Legal Compliance

```yaml
legal_compliance:
  allowed_licenses:
    - "Public Domain"
    - "CC0-1.0"
    - "CC-BY-4.0"
    - "CC-BY-SA-4.0"
  track_attribution: true
  generate_attribution_file: true
```

## Usage

### Local Execution

#### 1. Collect Data Only

```bash
python scripts/legal_data_collector.py
```

Output: `data/legal_sources/` with JSONL files and manifest

#### 2. Run Single Training Cycle

```bash
python scripts/continuous_learning_pipeline.py \
  --config configs/continuous_learning.yaml \
  --model-type text
```

#### 3. Run Continuous Learning (Scheduled)

```bash
python scripts/continuous_learning_pipeline.py \
  --config configs/continuous_learning.yaml \
  --model-type text \
  --continuous \
  --interval 24  # Run every 24 hours
```

#### 4. Skip Steps

```bash
# Skip data collection (use existing data)
python scripts/continuous_learning_pipeline.py \
  --config configs/continuous_learning.yaml \
  --model-type text \
  --skip-data-collection

# Skip training (evaluate existing model)
python scripts/continuous_learning_pipeline.py \
  --config configs/continuous_learning.yaml \
  --model-type text \
  --skip-training
```

#### 5. Deploy Improved Models

```bash
python scripts/continuous_learning_pipeline.py \
  --config configs/continuous_learning.yaml \
  --model-type text \
  --deploy
```

### Docker Execution

#### Build Image

```bash
cd docker
docker build -f Dockerfile.continuous_learning -t sarvanjna/continuous-learning ..
```

#### Run with Docker Compose

```bash
cd docker
docker-compose -f docker-compose.continuous_learning.yaml up
```

Services:
- **continuous-learning**: Main pipeline
- **tensorboard**: Monitoring (http://localhost:6006)
- **jupyter**: Interactive analysis (http://localhost:8888)

#### Run Single Container

```bash
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/outputs:/workspace/outputs \
  sarvanjna/continuous-learning
```

### Kubernetes Deployment

#### 1. Create Namespace

```bash
kubectl create namespace sarvanjna
```

#### 2. Deploy CronJob

```bash
kubectl apply -f k8s/continuous-learning-cronjob.yaml
```

This creates:
- ConfigMap with configuration
- CronJob (runs weekly on Sunday at 2 AM)
- PersistentVolumeClaims for data/outputs/models
- ServiceAccount and RBAC permissions

#### 3. Monitor Jobs

```bash
# List jobs
kubectl get cronjobs -n sarvanjna

# View job status
kubectl get jobs -n sarvanjna

# View pod logs
kubectl logs -n sarvanjna -l component=continuous-learning --tail=100
```

#### 4. Manual Trigger

```bash
kubectl create job -n sarvanjna \
  --from=cronjob/continuous-learning-text \
  manual-run-$(date +%s)
```

## Model Types

The pipeline supports four model types:

### Text (text)
- Model: Transformer-based language model
- Training: examples/train_text_model.py
- Metrics: Perplexity, BLEU, ROUGE
- Dataset: Wikipedia articles, Gutenberg books

### Image (image)
- Model: Latent Diffusion Model
- Training: examples/train_latent_diffusion.py
- Metrics: FID, Inception Score, CLIP Score
- Dataset: Wikimedia Commons images

### Video (video)
- Model: Text-to-Video diffusion
- Training: examples/train_text_to_video.py
- Metrics: FVD, CLIP Score
- Dataset: Video clips with captions

### Music (music)
- Model: MusicGen
- Training: examples/train_music.py
- Metrics: FAD, KL Divergence
- Dataset: Public domain music

## Output Structure

```
outputs/continuous_learning/
├── data/
│   ├── raw/                    # Raw collected data
│   └── processed/              # Preprocessed data
├── models/
│   ├── checkpoints/            # Model checkpoints
│   └── registry/               # Model metadata
├── logs/
│   ├── training/               # Training logs
│   ├── evaluation/             # Evaluation results
│   └── pipeline/               # Pipeline logs
└── pipeline_state.json         # Pipeline state
```

## Pipeline State

The pipeline tracks its state in `pipeline_state.json`:

```json
{
  "last_data_collection": "2024-01-15T10:30:00",
  "last_training": "2024-01-15T14:20:00",
  "current_version": 5,
  "best_metric": 0.85,
  "training_history": [
    {
      "timestamp": "2024-01-15T14:20:00",
      "checkpoint": "outputs/.../checkpoint_v5.ckpt",
      "metrics": {
        "primary_metric": 0.85,
        "perplexity": 25.3,
        "bleu": 0.45
      }
    }
  ]
}
```

## Legal Compliance

### Data Sources and Licenses

All training data is collected from legal sources with compatible licenses:

| Source | License | Use Case |
|--------|---------|----------|
| Wikipedia | CC-BY-SA 3.0 | Text data |
| Project Gutenberg | Public Domain | Books (pre-copyright) |
| Wikimedia Commons | CC0, CC-BY, CC-BY-SA | Images |
| Archive.org (pre-1928) | Public Domain | Historical content |

### Attribution Tracking

The system automatically tracks:
- Data source URLs
- License type and URL
- Collection date
- Attribution requirements

Output: `data/legal_sources/data_manifest.json`

```json
{
  "sources": [
    {
      "name": "Wikipedia - Machine Learning",
      "url": "https://en.wikipedia.org/wiki/Machine_learning",
      "license": "CC-BY-SA-3.0",
      "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
      "data_type": "text",
      "collection_date": "2024-01-15T10:30:00",
      "item_count": 150
    }
  ]
}
```

### Avoiding Copyright Issues

**✅ Legal Sources:**
- Wikipedia (with attribution)
- Project Gutenberg public domain books
- Pre-1928 Archive.org content
- CC-licensed Wikimedia images
- Open-source code (MIT, Apache)

**❌ Avoid:**
- Copyrighted books (post-1928)
- Commercial datasets without license
- Web scraping without permission
- Proprietary content

## Monitoring

### TensorBoard

```bash
tensorboard --logdir logs/training
```

View at: http://localhost:6006

Metrics:
- Training loss
- Evaluation metrics
- Learning rate schedule

### Weights & Biases (Optional)

Enable in config:

```yaml
monitoring:
  wandb:
    enabled: true
    project: "sarvanjna"
```

## Resource Requirements

### Minimum (Single GPU)
- GPU: NVIDIA RTX 3090 (24 GB VRAM)
- RAM: 32 GB
- Storage: 500 GB SSD
- CPU: 8 cores

### Recommended (Multi-GPU)
- GPU: 4x NVIDIA A100 (80 GB VRAM each)
- RAM: 256 GB
- Storage: 2 TB NVMe SSD
- CPU: 32 cores

### Cloud Instances
- AWS: p4d.24xlarge (8x A100)
- Azure: Standard_ND96asr_v4 (8x A100)
- GCP: a2-ultragpu-8g (8x A100)

## Troubleshooting

### Data Collection Fails

**Issue**: HTTP errors or rate limiting

**Solution**:
- Increase `request_delay` in config
- Check internet connection
- Verify API endpoints are accessible

### Out of Memory

**Issue**: CUDA OOM during training

**Solution**:
- Reduce batch_size
- Increase gradient_accumulation_steps
- Enable gradient checkpointing
- Use FSDP for large models

### Training Not Improving

**Issue**: Metrics not improving over versions

**Solution**:
- Verify data quality
- Check learning rate (may be too high/low)
- Ensure incremental learning is enabled
- Review data diversity

## Best Practices

1. **Start Small**: Begin with small datasets and short training runs
2. **Monitor Quality**: Check data quality before full training
3. **Version Control**: Track all config changes in Git
4. **Backup Models**: Keep checkpoints of best models
5. **Legal Compliance**: Always verify license compatibility
6. **Resource Limits**: Set max_training_hours to avoid runaway costs

## Future Enhancements

- [ ] Support for additional data sources (Common Crawl, OpenImages)
- [ ] Advanced quality filtering (NSFW detection, deduplication)
- [ ] Multi-modal training (text + image simultaneously)
- [ ] Federated learning support
- [ ] Active learning (prioritize valuable data)
- [ ] Automated hyperparameter tuning
- [ ] A/B testing for model comparison

## References

- [Wikipedia API](https://www.mediawiki.org/wiki/API:Main_page)
- [Project Gutenberg](https://www.gutenberg.org/)
- [Wikimedia Commons API](https://commons.wikimedia.org/wiki/Commons:API)
- [Creative Commons Licenses](https://creativecommons.org/licenses/)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)

## License

This continuous learning system is part of Sarvanjna and follows the same license.

All collected data retains its original license and attribution requirements.
