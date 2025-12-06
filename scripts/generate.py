"""Interactive text generation with trained model."""
import torch
import yaml
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ModelConfig, TransformerLM
from data import Tokenizer
from evaluation import interactive_generation
from training import CheckpointManager


def main():
    parser = argparse.ArgumentParser(description='Generate text with LLM')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_best.pt',
                        help='Path to checkpoint')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Single prompt (if not provided, enters interactive mode)')
    parser.add_argument('--max_tokens', type=int, default=100,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Nucleus sampling')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load configs
    with open('configs/model_config.yaml', 'r') as f:
        model_config_dict = yaml.safe_load(f)
    
    with open('configs/data_config.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Initialize model
    model_config = ModelConfig(**model_config_dict['model'])
    model = TransformerLM(model_config).to(device)
    
    # Load checkpoint
    checkpoint_manager = CheckpointManager()
    checkpoint_manager.load_checkpoint(
        args.checkpoint,
        model=model,
        device=device,
    )
    
    # Initialize tokenizer
    tokenizer = Tokenizer(
        tokenizer_type=data_config['dataset']['tokenizer_type'],
        tokenizer_path=data_config['dataset'].get('tokenizer_path'),
    )
    
    print(f"Model loaded from {args.checkpoint}")
    print(f"Parameters: {model.n_params:,}")
    
    # Single prompt or interactive mode
    if args.prompt:
        from evaluation import generate_text
        generated = generate_text(
            model,
            tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )
        print(f"\nGenerated:\n{generated}")
    else:
        interactive_generation(model, tokenizer, device=device)


if __name__ == '__main__':
    main()
