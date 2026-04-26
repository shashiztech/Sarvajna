"""
Inference script for Latent Diffusion Model (text-to-image generation).

Usage:
    python examples/inference_latent_diffusion.py \
        --checkpoint path/to/checkpoint.ckpt \
        --prompt "A beautiful sunset over mountains" \
        --output output.png \
        --num-steps 50 \
        --guidance-scale 7.5
"""

import argparse
import torch
from PIL import Image
import numpy as np

from sarvanjna.models.vision import LatentDiffusionModel, LatentDiffusionConfig
from sarvanjna.training.latent_diffusion_trainer import LatentDiffusionTrainer
from sarvanjna.preprocessing.tokenizer import SentencePieceTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Generate images with Latent Diffusion')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
    parser.add_argument('--negative-prompt', type=str, default='', help='Negative prompt')
    parser.add_argument('--output', type=str, default='output.png', help='Output image path')
    parser.add_argument('--tokenizer', type=str, default='models/tokenizer.model', help='Tokenizer path')
    parser.add_argument('--num-steps', type=int, default=50, help='Number of denoising steps')
    parser.add_argument('--guidance-scale', type=float, default=7.5, help='Classifier-free guidance scale')
    parser.add_argument('--height', type=int, default=512, help='Image height')
    parser.add_argument('--width', type=int, default=512, help='Image width')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--eta', type=float, default=0.0, help='DDIM eta (0.0 = deterministic)')
    parser.add_argument('--use-ema', action='store_true', help='Use EMA model weights')
    return parser.parse_args()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    # tensor: (3, H, W) in [-1, 1]
    array = tensor.cpu().numpy()
    array = (array + 1) / 2  # [-1, 1] -> [0, 1]
    array = (array * 255).clip(0, 255).astype(np.uint8)
    array = array.transpose(1, 2, 0)  # (H, W, 3)
    return Image.fromarray(array)


def main():
    args = parse_args()
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        generator = torch.Generator().manual_seed(args.seed)
    else:
        generator = None
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = SentencePieceTokenizer(args.tokenizer)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    trainer = LatentDiffusionTrainer.load_from_checkpoint(args.checkpoint)
    
    # Use EMA model if requested
    model = trainer.model_ema if (args.use_ema and hasattr(trainer, 'model_ema')) else trainer.model
    model = model.to(device)
    model.eval()
    
    # Tokenize prompt
    print(f"Prompt: {args.prompt}")
    tokens = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([tokens], device=device)
    
    # Generate image
    print(f"Generating image with {args.num_steps} steps...")
    with torch.no_grad():
        images = model.generate(
            input_ids=input_ids,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            eta=args.eta,
            generator=generator,
        )
    
    # Convert to PIL and save
    image = tensor_to_pil(images[0])
    image.save(args.output)
    print(f"Saved image to {args.output}")


if __name__ == '__main__':
    main()
