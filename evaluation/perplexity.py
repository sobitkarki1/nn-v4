"""Perplexity calculation for model evaluation."""
import torch
import math
from typing import Optional
from tqdm import tqdm

from ..models import TransformerLM


@torch.no_grad()
def calculate_perplexity(
    model: TransformerLM,
    dataloader: torch.utils.data.DataLoader,
    max_batches: Optional[int] = None,
    device: str = "cuda",
) -> float:
    """
    Calculate perplexity on a dataset.
    
    Perplexity = exp(average cross-entropy loss)
    Lower perplexity indicates better model performance.
    
    Args:
        model: Language model
        dataloader: DataLoader for evaluation data
        max_batches: Maximum number of batches to evaluate (None = all)
        device: Device to run evaluation on
        
    Returns:
        Perplexity score
    """
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Calculating perplexity", total=max_batches)
    
    for batch in pbar:
        if max_batches and num_batches >= max_batches:
            break
        
        # Move to device
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']
        
        # Accumulate loss
        batch_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        num_batches += 1
        
        # Update progress bar
        current_ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else 0
        pbar.set_postfix({'perplexity': f'{current_ppl:.2f}'})
    
    # Calculate final perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)
    
    return perplexity


@torch.no_grad()
def evaluate_model(
    model: TransformerLM,
    dataloader: torch.utils.data.DataLoader,
    max_batches: Optional[int] = None,
    device: str = "cuda",
) -> dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Language model
        dataloader: DataLoader for evaluation data
        max_batches: Maximum number of batches to evaluate
        device: Device to run evaluation on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Evaluating", total=max_batches):
        if max_batches and num_batches >= max_batches:
            break
        
        # Move to device
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']
        
        # Accumulate
        batch_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        num_batches += 1
    
    # Calculate metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'num_batches': num_batches,
        'num_tokens': total_tokens,
    }
