"""Multi-head self-attention implementation with optional Flash Attention."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from .embeddings import RotaryPositionalEmbedding


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with optional Flash Attention.
    
    Supports:
    - Causal masking for autoregressive generation
    - Rotary Position Embeddings (RoPE)
    - Flash Attention 2 for memory efficiency
    - Optional bias terms
    """
    
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        context_length: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_flash_attn: bool = False,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.context_length = context_length
        self.use_flash_attn = use_flash_attn
        self.use_rope = use_rope
        
        # QKV projection
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(n_embd, n_embd, bias=bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Rotary embeddings
        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                dim=self.head_dim,
                max_seq_len=context_length,
                theta=rope_theta
            )
        
        # Causal mask (if not using Flash Attention)
        if not use_flash_attn:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(context_length, context_length))
                .view(1, 1, context_length, context_length),
                persistent=False
            )
        
        # Try to import Flash Attention
        if use_flash_attn:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
            except ImportError:
                print("Warning: flash_attn not installed, falling back to standard attention")
                self.use_flash_attn = False
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, n_embd]
            attention_mask: Optional attention mask [batch, seq_len]
            
        Returns:
            Output tensor [batch, seq_len, n_embd]
        """
        batch_size, seq_len, _ = x.shape
        
        # QKV projection
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape to [batch, n_head, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        if self.use_rope:
            q, k = self.rope(q, k)
        
        # Compute attention
        if self.use_flash_attn and self.training:
            # Flash Attention (training only, requires specific tensor layout)
            attn_output = self._flash_attention(q, k, v)
        else:
            # Standard scaled dot-product attention
            attn_output = self._standard_attention(q, k, v, attention_mask)
        
        # Reshape back to [batch, seq_len, n_embd]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)
        
        return output
    
    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard scaled dot-product attention."""
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        seq_len = q.shape[2]
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output
    
    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Flash Attention 2 (memory efficient)."""
        # Flash Attention expects [batch, seq_len, n_head, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Call Flash Attention (causal=True for autoregressive)
        attn_output = self.flash_attn_func(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            causal=True,
        )
        
        # Transpose back to [batch, n_head, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2)
        
        return attn_output
