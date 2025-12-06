"""DataLoader utilities."""
import torch
from torch.utils.data import DataLoader
from typing import Optional

from .dataset import TextDataset
from .tokenizer import Tokenizer


def get_dataloader(
    dataset_name: str,
    dataset_path: str,
    tokenizer: Tokenizer,
    split: str = "train",
    max_length: int = 2048,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    streaming: bool = True,
    shuffle_buffer_size: int = 10000,
    cache_dir: Optional[str] = None,
) -> DataLoader:
    """
    Create a DataLoader for language model training.
    
    Args:
        dataset_name: Name of the dataset
        dataset_path: Path to dataset or HuggingFace dataset name
        tokenizer: Tokenizer instance
        split: Dataset split ('train', 'validation', etc.)
        max_length: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        streaming: Whether to use streaming mode
        shuffle_buffer_size: Buffer size for shuffling (streaming mode)
        cache_dir: Cache directory for datasets
        
    Returns:
        DataLoader instance
    """
    dataset = TextDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        split=split,
        max_length=max_length,
        streaming=streaming,
        shuffle_buffer_size=shuffle_buffer_size,
        cache_dir=cache_dir,
    )
    
    # For streaming datasets, we don't shuffle in DataLoader
    # (shuffling is done in the dataset itself)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers if streaming else 0,  # Use 0 workers for non-streaming
        pin_memory=pin_memory,
        shuffle=False,  # Don't shuffle in DataLoader for streaming
    )
    
    return dataloader


def collate_fn(batch):
    """
    Collate function for batching.
    
    Args:
        batch: List of examples
        
    Returns:
        Batched tensors
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'labels': labels,
    }
