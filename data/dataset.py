"""Dataset implementation for language model training."""
import torch
from torch.utils.data import IterableDataset, Dataset
from datasets import load_dataset
from typing import Optional, Iterator, Dict
import numpy as np

from .tokenizer import Tokenizer


class TextDataset(IterableDataset):
    """
    Streaming text dataset for large-scale language model training.
    
    Supports:
    - Streaming mode for datasets that don't fit in memory
    - Automatic tokenization and chunking
    - Document-level shuffling
    """
    
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        tokenizer: Tokenizer,
        split: str = "train",
        max_length: int = 2048,
        streaming: bool = True,
        shuffle_buffer_size: int = 10000,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            dataset_name: Name of the dataset
            dataset_path: Path to dataset or HuggingFace dataset name
            tokenizer: Tokenizer instance
            split: Dataset split ('train', 'validation', etc.)
            max_length: Maximum sequence length
            streaming: Whether to use streaming mode
            shuffle_buffer_size: Buffer size for shuffling (streaming mode)
            cache_dir: Cache directory for datasets
        """
        super().__init__()
        
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.streaming = streaming
        
        # Load dataset
        try:
            if dataset_name == "the_pile":
                # The Pile dataset
                self.dataset = load_dataset(
                    "EleutherAI/pile",
                    split=split,
                    streaming=streaming,
                    cache_dir=cache_dir,
                )
            else:
                # Generic HuggingFace dataset
                self.dataset = load_dataset(
                    dataset_path,
                    split=split,
                    streaming=streaming,
                    cache_dir=cache_dir,
                )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print(f"Using dummy dataset for testing")
            self.dataset = self._create_dummy_dataset()
        
        # Shuffle if streaming
        if streaming and split == "train":
            self.dataset = self.dataset.shuffle(buffer_size=shuffle_buffer_size, seed=42)
    
    def _create_dummy_dataset(self):
        """Create a dummy dataset for testing."""
        dummy_data = {
            "text": [
                "This is a sample text for testing the language model. " * 100
                for _ in range(1000)
            ]
        }
        from datasets import Dataset as HFDataset
        return HFDataset.from_dict(dummy_data)
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over dataset, yielding tokenized chunks.
        
        Yields:
            Dictionary with 'input_ids' and 'labels'
        """
        buffer = []
        
        for example in self.dataset:
            # Get text from example
            text = example.get('text', example.get('content', ''))
            
            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            
            # Yield chunks of max_length
            while len(buffer) >= self.max_length + 1:
                # Extract chunk
                chunk = buffer[:self.max_length + 1]
                buffer = buffer[self.max_length:]
                
                # Create input_ids and labels
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                
                yield {
                    'input_ids': input_ids,
                    'labels': labels,
                }
    
    def __len__(self):
        """
        Return approximate length (not available for streaming datasets).
        
        Note: This is only approximate and may not be accurate for streaming datasets.
        """
        if self.streaming:
            return 1_000_000  # Arbitrary large number for streaming
        else:
            return len(self.dataset)


class SimpleTextDataset(Dataset):
    """
    Simple non-streaming text dataset for small datasets.
    
    Loads entire dataset into memory and pre-tokenizes.
    """
    
    def __init__(
        self,
        texts: list,
        tokenizer: Tokenizer,
        max_length: int = 2048,
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of text strings
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize all texts
        self.examples = []
        buffer = []
        
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            
            # Create chunks
            while len(buffer) >= max_length + 1:
                chunk = buffer[:max_length + 1]
                buffer = buffer[max_length:]
                
                self.examples.append({
                    'input_ids': torch.tensor(chunk[:-1], dtype=torch.long),
                    'labels': torch.tensor(chunk[1:], dtype=torch.long),
                })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
