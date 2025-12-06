"""Utility functions for logging and monitoring."""
import os
from typing import Optional


def setup_logging(config: dict):
    """
    Set up logging backends (TensorBoard, Weights & Biases).
    
    Args:
        config: Logging configuration dictionary
    """
    loggers = {}
    
    # TensorBoard
    if config.get('use_tensorboard', True):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tensorboard_dir = config.get('tensorboard_dir', 'logs/tensorboard')
            os.makedirs(tensorboard_dir, exist_ok=True)
            loggers['tensorboard'] = SummaryWriter(tensorboard_dir)
            print(f"TensorBoard logging enabled: {tensorboard_dir}")
        except ImportError:
            print("Warning: TensorBoard not available")
    
    # Weights & Biases
    if config.get('use_wandb', False):
        try:
            import wandb
            wandb.init(
                project=config.get('wandb_project', 'llm-training'),
                entity=config.get('wandb_entity', None),
                config=config,
            )
            loggers['wandb'] = wandb
            print("Weights & Biases logging enabled")
        except ImportError:
            print("Warning: wandb not installed")
    
    return loggers


def log_metrics(loggers: dict, metrics: dict, step: int):
    """
    Log metrics to all enabled logging backends.
    
    Args:
        loggers: Dictionary of logger instances
        metrics: Dictionary of metrics to log
        step: Current training step
    """
    # TensorBoard
    if 'tensorboard' in loggers:
        writer = loggers['tensorboard']
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(key, value, step)
    
    # Weights & Biases
    if 'wandb' in loggers:
        loggers['wandb'].log(metrics, step=step)


def close_loggers(loggers: dict):
    """
    Close all logging backends.
    
    Args:
        loggers: Dictionary of logger instances
    """
    if 'tensorboard' in loggers:
        loggers['tensorboard'].close()
    
    if 'wandb' in loggers:
        loggers['wandb'].finish()
