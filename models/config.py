"""Model configuration dataclass."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the transformer model."""
    
    # Architecture
    vocab_size: int = 50257
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1536
    context_length: int = 2048
    dropout: float = 0.0
    bias: bool = False
    
    # Positional encoding
    use_rope: bool = True
    rope_theta: float = 10000.0
    
    # Attention
    use_flash_attn: bool = False
    
    # Normalization
    layer_norm_epsilon: float = 1e-5
    
    # Feed-forward network
    ffn_hidden_mult: int = 4
    
    @property
    def ffn_hidden_size(self) -> int:
        """Calculate FFN hidden size."""
        return self.n_embd * self.ffn_hidden_mult
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.n_layer > 0, "n_layer must be positive"
