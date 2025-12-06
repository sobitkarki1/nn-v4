"""Optimizer setup and configuration."""
import torch
from torch.optim import AdamW
from typing import Optional


def get_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 6e-4,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    use_8bit: bool = False,
    fused: bool = True,
) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer with weight decay.
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        beta1: Adam beta1
        beta2: Adam beta2
        eps: Adam epsilon
        use_8bit: Use 8-bit optimizer (bitsandbytes)
        fused: Use fused AdamW kernel (faster)
        
    Returns:
        Optimizer instance
    """
    # Separate parameters that should and shouldn't have weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # No weight decay for biases and layer norms
        if param.dim() < 2 or 'ln' in name or 'bias' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    
    print(f"Parameters with weight decay: {sum(p.numel() for p in decay_params):,}")
    print(f"Parameters without weight decay: {sum(p.numel() for p in no_decay_params):,}")
    
    # Create optimizer
    if use_8bit:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                optim_groups,
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=eps,
            )
            print("Using 8-bit AdamW optimizer")
        except ImportError:
            print("Warning: bitsandbytes not installed, using standard AdamW")
            use_8bit = False
    
    if not use_8bit:
        optimizer = AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            fused=fused and torch.cuda.is_available(),
        )
        print(f"Using AdamW optimizer (fused={fused and torch.cuda.is_available()})")
    
    return optimizer
