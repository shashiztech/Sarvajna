# Continuous Learning Quick Start

Get your automated training pipeline running in 5 minutes!

## Prerequisites

- Python 3.8+
- PyTorch 2.0+ with CUDA (for GPU)
- 32+ GB RAM
- 500+ GB storage
- Internet connection

## Step 1: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Step 2: Verify Setup

Run the setup script to check everything:

```bash
python scripts/setup_continuous_learning.py
```

This will:
- ✅ Check Python version
- ✅ Verify dependencies
- ✅ Check GPU availability
- ✅ Create required directories
- ✅ Test data source connections

## Step 3: Collect Data

Collect training data from legal sources:

```bash
python scripts/legal_data_collector.py
```

This collects data from:
- **Wikipedia**: CC-BY-SA 3.0 licensed articles
- **Project Gutenberg**: Public domain books
- **Wikimedia Commons**: CC-licensed images

Output: `data/legal_sources/` with JSONL files and manifest

**Expected output:**
```
Collected 150 articles from Wikipedia
Collected 50 books from Project Gutenberg
Collected 5000 image URLs from Wikimedia Commons
Data manifest saved to: data/legal_sources/data_manifest.json
```

## Step 4: Run Training Pipeline

### Option A: Single Training Run

Train once on the collected data:

```bash
python scripts/continuous_learning_pipeline.py \
  --config configs/continuous_learning.yaml \
  --model-type text
```

**Expected duration:** 2-8 hours (depending on GPU)

### Option B: Continuous Learning

Run automatically every 24 hours:

```bash
python scripts/continuous_learning_pipeline.py \
  --config configs/continuous_learning.yaml \
  --model-type text \
  --continuous \
  --interval 24
```

This will:
1. Collect new data
2. Preprocess data
3. Train model incrementally
4. Evaluate performance
5. Version and register model
6. Wait 24 hours
7. Repeat

## Step 5: Monitor Training

### TensorBoard

```bash
tensorboard --logdir logs/training
```

Open http://localhost:6006 to view:
- Training loss curves
- Evaluation metrics
- Learning rate schedule

### Pipeline State

Check pipeline state:

```bash
cat outputs/continuous_learning/pipeline_state.json
```

Shows:
- Current model version
- Best metric achieved
- Last data collection/training times
- Training history

## Advanced Usage

### Deploy Improved Models

Automatically deploy when model improves:

```bash
python scripts/continuous_learning_pipeline.py \
  --config configs/continuous_learning.yaml \
  --model-type text \
  --deploy
```

Deployed models saved to: `deployments/text/model_latest.ckpt`

### Skip Steps

Skip data collection (use existing data):

```bash
python scripts/continuous_learning_pipeline.py \
  --config configs/continuous_learning.yaml \
  --model-type text \
  --skip-data-collection
```

Skip training (evaluate only):

```bash
python scripts/continuous_learning_pipeline.py \
  --config configs/continuous_learning.yaml \
  --model-type text \
  --skip-training
```

### Different Model Types

Train image generation model:

```bash
python scripts/continuous_learning_pipeline.py \
  --config configs/continuous_learning.yaml \
  --model-type image
```

Available types: `text`, `image`, `video`, `music`

## Docker Usage

### Build and Run

```bash
# Build image
cd docker
docker build -f Dockerfile.continuous_learning -t sarvanjna/continuous-learning ..

# Run with Docker Compose
docker-compose -f docker-compose.continuous_learning.yaml up
```

Services:
- **continuous-learning**: Training pipeline
- **tensorboard**: Monitoring (http://localhost:6006)
- **jupyter**: Interactive analysis (http://localhost:8888)

## Kubernetes Deployment

### Deploy to Cluster

```bash
# Create namespace
kubectl create namespace sarvanjna

# Deploy CronJob (runs weekly)
kubectl apply -f k8s/continuous-learning-cronjob.yaml

# Monitor
kubectl get cronjobs -n sarvanjna
kubectl logs -n sarvanjna -l component=continuous-learning
```

### Manual Trigger

```bash
kubectl create job -n sarvanjna \
  --from=cronjob/continuous-learning-text \
  manual-run-$(date +%s)
```

## Configuration

Edit `configs/continuous_learning.yaml` to customize:

### Data Collection

```yaml
data_collection:
  wikipedia:
    max_articles: 1000  # Increase for more data
    categories: ["Your", "Topics"]
```

### Training

```yaml
training:
  text:
    batch_size: 32  # Adjust for your GPU
    max_epochs: 10
    learning_rate: 1.0e-4
```

### Schedule

```yaml
continuous_learning:
  enabled: true
  interval_hours: 24  # How often to train
```

## Troubleshooting

### Out of Memory

**Error**: `CUDA out of memory`

**Fix**: Reduce batch size in config:
```yaml
training:
  text:
    batch_size: 16  # Was 32
    gradient_accumulation_steps: 8  # Was 4
```

### Data Collection Fails

**Error**: HTTP errors or timeouts

**Fix**: Increase delay in `scripts/legal_data_collector.py`:
```python
self.request_delay = 2.0  # Was 1.0
```

### No GPU Detected

**Error**: `No GPU available`

**Fix**: 
1. Check CUDA installation: `nvidia-smi`
2. Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### Training Not Improving

**Issue**: Metrics plateau or decrease

**Fix**:
1. Check data quality: Review `data/legal_sources/data_manifest.json`
2. Reduce learning rate: Set `learning_rate: 5.0e-5` in config
3. Increase data diversity: Add more Wikipedia categories

## Expected Results

### After First Training Run

- **Model version**: 1
- **Text model perplexity**: ~40-60 (untrained: ~100+)
- **Output quality**: Basic coherence, some grammatical errors
- **Time**: 2-8 hours (depending on data size and GPU)

### After 10 Training Runs

- **Model version**: 10
- **Text model perplexity**: ~20-30
- **Output quality**: Good coherence, fluent text
- **Best practices**: Model shows consistent improvement

### Production Ready (50+ runs)

- **Model version**: 50+
- **Text model perplexity**: ~10-20
- **Output quality**: High-quality, contextually appropriate
- **Deployment**: Ready for production use

## Data Sources Summary

| Source | License | Data Type | Typical Yield |
|--------|---------|-----------|---------------|
| Wikipedia | CC-BY-SA 3.0 | Text | 100-1000 articles |
| Project Gutenberg | Public Domain | Books | 50-100 books |
| Wikimedia Commons | Various CC | Images | 5000-10000 images |
| Archive.org (pre-1928) | Public Domain | Historical | Manually curated |

## Legal Compliance

✅ **Safe to use:**
- Wikipedia (with attribution)
- Project Gutenberg public domain books
- Pre-1928 Archive.org content
- CC-licensed Wikimedia images

❌ **Avoid:**
- Copyrighted books (post-1928)
- Web scraping without permission
- Proprietary datasets
- Copyright-protected images

All data sources are tracked in: `data/legal_sources/data_manifest.json`

## Next Steps

1. ✅ Complete this quick start
2. 📖 Read full documentation: [docs/CONTINUOUS_LEARNING.md](CONTINUOUS_LEARNING.md)
3. 🔧 Customize configuration for your needs
4. 🚀 Set up continuous learning schedule
5. 📊 Monitor metrics and iterate

## Getting Help

- **Documentation**: See `docs/CONTINUOUS_LEARNING.md`
- **Issues**: Check GitHub Issues
- **Examples**: See `examples/` directory
- **Config**: Review `configs/continuous_learning.yaml`

## Summary Commands

```bash
# 1. Setup
python scripts/setup_continuous_learning.py

# 2. Collect data
python scripts/legal_data_collector.py

# 3. Train once
python scripts/continuous_learning_pipeline.py \
  --config configs/continuous_learning.yaml \
  --model-type text

# 4. Monitor
tensorboard --logdir logs/training

# 5. Deploy
python scripts/continuous_learning_pipeline.py \
  --config configs/continuous_learning.yaml \
  --model-type text \
  --deploy
```

## Success Indicators

✅ Setup script passes all checks  
✅ Data collection completes without errors  
✅ Training runs without OOM errors  
✅ Metrics improve over versions  
✅ Pipeline state updates correctly  
✅ TensorBoard shows training curves  
✅ Models deploy successfully  

🎉 **You're ready to go! Happy training!** 🎉
