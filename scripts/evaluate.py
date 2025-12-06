"""Evaluate trained model."""
import torch
import yaml
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ModelConfig, TransformerLM
from data import Tokenizer, get_dataloader
from evaluation import evaluate_model
from training import CheckpointManager


def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_best.pt',
                        help='Path to checkpoint')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load configs
    with open('configs/model_config.yaml', 'r') as f:
        model_config_dict = yaml.safe_load(f)
    
    with open('configs/training_config.yaml', 'r') as f:
        train_config = yaml.safe_load(f)
    
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
    
    # Load validation data
    val_loader = get_dataloader(
        dataset_name=data_config['dataset']['name'],
        dataset_path=data_config['dataset']['path'],
        tokenizer=tokenizer,
        split=data_config['dataset']['val_split'],
        max_length=data_config['dataset']['max_length'],
        batch_size=train_config['training']['batch_size'],
        streaming=data_config['dataset']['streaming'],
    )
    
    # Evaluate
    print(f"\nEvaluating model from {args.checkpoint}")
    results = evaluate_model(
        model,
        val_loader,
        max_batches=train_config['logging']['eval_batches'],
        device=device,
    )
    
    print(f"\nResults:")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Perplexity: {results['perplexity']:.2f}")
    print(f"Tokens evaluated: {results['num_tokens']:,}")


if __name__ == '__main__':
    main()
