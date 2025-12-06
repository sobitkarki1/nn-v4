"""Learning rate scheduler implementations."""
import math
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    num_cycles: float = 0.5,
):
    """
    Create a learning rate scheduler with linear warmup and cosine decay.
    
    Args:
        optimizer: Optimizer instance
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as ratio of initial LR
        num_cycles: Number of cosine cycles (0.5 = half cycle)
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        
        # Scale to min_lr_ratio
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
):
    """
    Create a learning rate scheduler with linear warmup and linear decay.
    
    Args:
        optimizer: Optimizer instance
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as ratio of initial LR
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Linear decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)
    
    return LambdaLR(optimizer, lr_lambda)


def get_constant_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
):
    """
    Create a learning rate scheduler with linear warmup and constant LR.
    
    Args:
        optimizer: Optimizer instance
        num_warmup_steps: Number of warmup steps
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda)


def get_scheduler(
    optimizer,
    scheduler_type: str = "cosine",
    num_warmup_steps: int = 2000,
    num_training_steps: int = 100000,
    min_lr_ratio: float = 0.1,
):
    """
    Get learning rate scheduler by type.
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler ('cosine', 'linear', 'constant')
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate ratio
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, min_lr_ratio
        )
    elif scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, min_lr_ratio
        )
    elif scheduler_type == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
