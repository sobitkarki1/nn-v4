"""Feed-forward network (FFN) implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    Standard transformer FFN: Linear -> GELU -> Linear
    """
    
    def __init__(
        self,
        n_embd: int,
        ffn_hidden_size: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(n_embd, ffn_hidden_size, bias=bias)
        self.fc2 = nn.Linear(ffn_hidden_size, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, n_embd]
            
        Returns:
            Output tensor [batch, seq_len, n_embd]
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SwiGLU(nn.Module):
    """
    SwiGLU activation (used in LLaMA, PaLM).
    
    SwiGLU(x) = Swish(xW) ⊙ xV
    where Swish(x) = x * sigmoid(x)
    
    Note: Requires wider hidden dimension to maintain parameter count.
    """
    
    def __init__(
        self,
        n_embd: int,
        ffn_hidden_size: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        
        # SwiGLU uses gating, so we need two projections up
        self.w1 = nn.Linear(n_embd, ffn_hidden_size, bias=bias)
        self.w2 = nn.Linear(n_embd, ffn_hidden_size, bias=bias)
        self.w3 = nn.Linear(ffn_hidden_size, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, n_embd]
            
        Returns:
            Output tensor [batch, seq_len, n_embd]
        """
        # SwiGLU activation
        x1 = F.silu(self.w1(x))  # Swish/SiLU activation
        x2 = self.w2(x)
        x = x1 * x2  # Element-wise multiplication (gating)
        x = self.w3(x)
        x = self.dropout(x)
        return x
