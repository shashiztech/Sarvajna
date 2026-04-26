"""
Quick demo to test the Sarvanjna AI platform.
Train a tokenizer and run a small model.
"""

import torch
from pathlib import Path
import tempfile

from sarvanjna.preprocessing.tokenizer import SentencePieceTokenizer
from sarvanjna.models.text.transformer import TransformerConfig
from sarvanjna.models.text.text_to_text import TextToTextModel

print("=" * 70)
print("Sarvanjna AI Platform - Quick Demo")
print("=" * 70)

# Step 1: Create a sample corpus
print("\n[1/4] Creating sample corpus...")
with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
    corpus_file = Path(f.name)
    sentences = [
        "Hello world, how are you today?",
        "This is a test of the text generation system.",
        "Machine learning is transforming artificial intelligence.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models use neural networks for pattern recognition.",
        "Python is a popular programming language for data science.",
        "Transformers revolutionized natural language understanding.",
        "Text generation models can create human-like responses."
    ]
    f.write("\n".join(sentences * 20))  # Repeat for more data

print(f"   ✓ Corpus created: {corpus_file}")

# Step 2: Train tokenizer
print("\n[2/4] Training SentencePiece tokenizer...")
tokenizer = SentencePieceTokenizer(vocab_size=100, model_type="bpe")
with tempfile.TemporaryDirectory() as tmpdir:
    model_prefix = str(Path(tmpdir) / "demo_tokenizer")
    tokenizer.train([corpus_file], model_prefix, num_threads=1)
    print(f"   ✓ Tokenizer trained with vocab size: {tokenizer.vocab_size}")
    
    # Test tokenization
    test_text = "Hello world machine learning"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded.ids)
    print(f"   ✓ Test: '{test_text}' → {len(encoded.ids)} tokens → '{decoded}'")

# Step 3: Create a small model
print("\n[3/4] Creating Text-to-Text model...")
config = TransformerConfig(
    vocab_size=100,
    d_model=128,       # Small for demo
    n_heads=4,
    n_layers=2,
    d_ff=512,
    max_seq_length=50,
)
model = TextToTextModel(config)
num_params = model.get_num_params()
print(f"   ✓ Model initialized")
print(f"   ✓ Parameters: {num_params:,} (~{num_params/1e6:.2f}M)")
print(f"   ✓ Architecture: {config.n_layers} layers, {config.d_model} dims, {config.n_heads} heads")

# Step 4: Test forward pass
print("\n[4/4] Testing model forward pass...")
batch_size = 2
seq_len = 10

input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))
decoder_input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))
labels = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))

model.eval()
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        labels=labels
    )
    
print(f"   ✓ Forward pass successful")
print(f"   ✓ Output shape: {outputs['logits'].shape}")
print(f"   ✓ Loss: {outputs['loss'].item():.4f}")

# Test generation
print("\n[5/5] Testing text generation...")
input_ids = torch.randint(0, tokenizer.vocab_size, (1, 5))
generated = model.generate(input_ids=input_ids, max_length=15)
print(f"   ✓ Generated {generated.shape[1]} tokens from {input_ids.shape[1]} input tokens")

print("\n" + "=" * 70)
print("✅ All systems operational! Sarvanjna AI platform is working.")
print("=" * 70)
print("\nNext steps:")
print("  • Train a tokenizer: python examples/train_tokenizer.py")
print("  • Train a model:     python examples/train_text_model.py")
print("  • Run inference:     python examples/inference_text.py")
print("  • Read the guide:    QUICKSTART.md")

# Cleanup
corpus_file.unlink()
