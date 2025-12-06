"""Tokenizer wrapper for text preprocessing."""
import torch
from transformers import AutoTokenizer
from typing import Union, List, Optional


class Tokenizer:
    """
    Tokenizer wrapper supporting various tokenizer types.
    
    Supports:
    - GPT-2 BPE tokenizer (default)
    - Custom tokenizers from HuggingFace
    - Local tokenizer files
    """
    
    def __init__(
        self,
        tokenizer_type: str = "gpt2",
        tokenizer_path: Optional[str] = None,
    ):
        """
        Initialize tokenizer.
        
        Args:
            tokenizer_type: Type of tokenizer ('gpt2', 'custom', etc.)
            tokenizer_path: Path to custom tokenizer or HF model name
        """
        self.tokenizer_type = tokenizer_type
        
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        elif tokenizer_type == "gpt2":
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.vocab_size = len(self.tokenizer)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id or self.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
    
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> Union[List[int], torch.Tensor]:
        """
        Encode text to token IDs.
        
        Args:
            text: Text or list of texts to encode
            add_special_tokens: Whether to add special tokens
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            
        Returns:
            Token IDs
        """
        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=None,
        )
        
        if isinstance(text, str):
            return encoded['input_ids']
        else:
            return encoded['input_ids']
    
    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def batch_decode(
        self,
        token_ids: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode batch of token IDs to text.
        
        Args:
            token_ids: Batch of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded texts
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
