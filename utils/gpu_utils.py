"""GPU memory monitoring utilities."""
import torch
from typing import Dict


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get GPU memory usage information.
    
    Returns:
        Dictionary with memory statistics in GB
    """
    if not torch.cuda.is_available():
        return {}
    
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    
    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'max_allocated_gb': max_allocated,
    }


def print_gpu_memory():
    """Print GPU memory usage."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    info = get_gpu_memory_info()
    print(f"GPU Memory - Allocated: {info['allocated_gb']:.2f}GB, "
          f"Reserved: {info['reserved_gb']:.2f}GB, "
          f"Max: {info['max_allocated_gb']:.2f}GB")


def reset_peak_memory_stats():
    """Reset peak memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
