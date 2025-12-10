"""Rotary Position Embeddings (RoPE) implementation."""
import torch
import torch.nn as nn
from typing import Tuple


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) as used in models like GPT-NeoX, LLaMA.
    
    Applies rotary embeddings to query and key tensors in attention.
    More efficient than absolute positional embeddings and enables length extrapolation.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos and sin for max sequence length
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for given sequence length."""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        
        # Create rotation matrix [seq_len, dim]
        # We duplicate to match full head dimension
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_cached_len = seq_len
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key.
        
        Args:
            q: Query tensor [batch, n_head, seq_len, head_dim]
            k: Key tensor [batch, n_head, seq_len, head_dim]
            
        Returns:
            Rotated q and k tensors
        """
        seq_len = q.shape[2]
        
        # Extend cache if needed
        if seq_len > self.max_cached_len:
            self._build_cache(seq_len)
        
        # Get cached cos/sin for current sequence
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        # Apply rotation
        q_rot = self._apply_rotary_emb(q, cos, sin)
        k_rot = self._apply_rotary_emb(k, cos, sin)
        
        return q_rot, k_rot
    
    def _apply_rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embeddings to input tensor."""
        # x: [batch, n_head, seq_len, head_dim]
        # cos, sin: [seq_len, head_dim]
        
        # Reshape for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Split x into two halves for rotation
        head_dim = x.shape[-1]
        x1 = x[..., :head_dim // 2]
        x2 = x[..., head_dim // 2:]
        
        # Get corresponding cos/sin values for each half
        cos_half = cos[..., :head_dim // 2]
        sin_half = sin[..., :head_dim // 2]
        
        # Rotation formula: [x1, x2] @ [[cos, -sin], [sin, cos]]
        rotated = torch.cat([
            x1 * cos_half - x2 * sin_half,
            x1 * sin_half + x2 * cos_half
        ], dim=-1)
        
        return rotated
