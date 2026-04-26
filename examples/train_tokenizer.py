"""
Example script for training a SentencePiece tokenizer.

Usage:
    python examples/train_tokenizer.py --input data/corpus.txt --output models/tokenizer
"""

import argparse
from pathlib import Path

from sarvanjna.preprocessing.tokenizer import SentencePieceTokenizer


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer")
    parser.add_argument("--input", type=str, required=True, nargs="+", help="Input text files")
    parser.add_argument("--output", type=str, required=True, help="Output model prefix")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--model_type", type=str, default="unigram", choices=["unigram", "bpe"], help="Model type")
    parser.add_argument("--character_coverage", type=float, default=0.9995, help="Character coverage")
    parser.add_argument("--num_threads", type=int, default=16, help="Number of threads")
    
    args = parser.parse_args()
    
    # Validate input files
    input_files = [Path(f) for f in args.input]
    for f in input_files:
        if not f.exists():
            raise FileNotFoundError(f"Input file not found: {f}")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Training SentencePiece tokenizer...")
    print(f"  Input files: {len(input_files)}")
    print(f"  Vocab size: {args.vocab_size}")
    print(f"  Model type: {args.model_type}")
    print(f"  Output: {args.output}")
    
    # Train tokenizer
    tokenizer = SentencePieceTokenizer(
        vocab_size=args.vocab_size,
        model_type=args.model_type,
    )
    
    tokenizer.train(
        input_files=input_files,
        model_prefix=args.output,
        character_coverage=args.character_coverage,
        num_threads=args.num_threads,
    )
    
    # Test tokenizer
    print("\nTesting tokenizer...")
    test_text = "This is a test sentence for the tokenizer."
    encoded = tokenizer.encode(test_text, add_bos=True, add_eos=True)
    decoded = tokenizer.decode(encoded.ids)
    
    print(f"  Original: {test_text}")
    print(f"  Tokens: {encoded.tokens}")
    print(f"  IDs: {encoded.ids}")
    print(f"  Decoded: {decoded}")
    
    print(f"\n✓ Tokenizer saved to: {args.output}.model")


if __name__ == "__main__":
    main()
