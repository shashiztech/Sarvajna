"""
Example inference script for Text-to-Text model.

Usage:
    python examples/inference_text.py --checkpoint checkpoints/model.ckpt --tokenizer models/tokenizer.model
"""

import argparse
from pathlib import Path
import torch

from sarvanjna.preprocessing.tokenizer import SentencePieceTokenizer
from sarvanjna.training.text_trainer import TextToTextTrainer


def main():
    parser = argparse.ArgumentParser(description="Text-to-Text inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer model path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_length", type=int, default=100, help="Max generation length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling instead of greedy")
    
    args = parser.parse_args()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = SentencePieceTokenizer(model_path=args.tokenizer)
    
    # Load model
    print("Loading model...")
    model = TextToTextTrainer.load_from_checkpoint(args.checkpoint)
    model = model.to(args.device)
    model.eval()
    
    # Print model info
    info = model.get_model_info()
    print(f"\nModel loaded:")
    print(f"  Parameters: {info['total_params']:,}")
    print(f"  Device: {args.device}")
    
    # Interactive loop
    print("\n" + "="*60)
    print("Text-to-Text Model - Interactive Mode")
    print("="*60)
    print("Enter text to generate responses (or 'quit' to exit)\n")
    
    while True:
        # Get input
        try:
            input_text = input("Input: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if not input_text or input_text.lower() == "quit":
            print("Exiting...")
            break
        
        # Encode input
        encoded = tokenizer.encode(input_text, add_bos=True)
        input_ids = torch.tensor([encoded.ids]).to(args.device)
        attention_mask = torch.tensor([encoded.attention_mask]).to(args.device)
        
        # Generate
        print("Generating...")
        with torch.no_grad():
            output_ids = model.generate_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=args.do_sample,
            )
        
        # Decode output
        output_text = tokenizer.decode(output_ids[0].cpu().tolist())
        
        print(f"Output: {output_text}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
