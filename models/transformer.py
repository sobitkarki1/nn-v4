"""Main transformer model implementation."""
import torch
import torch.nn as nn
from typing import Optional
import torch.utils.checkpoint as checkpoint

from .config import ModelConfig
from .attention import MultiHeadAttention
from .ffn import FeedForward


class TransformerBlock(nn.Module):
    """Single transformer block with attention and FFN."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        # Layer norms (pre-norm architecture)
        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Attention
        self.attn = MultiHeadAttention(
            n_embd=config.n_embd,
            n_head=config.n_head,
            context_length=config.context_length,
            dropout=config.dropout,
            bias=config.bias,
            use_flash_attn=config.use_flash_attn,
            use_rope=config.use_rope,
            rope_theta=config.rope_theta,
        )
        
        # Feed-forward
        self.ffn = FeedForward(
            n_embd=config.n_embd,
            ffn_hidden_size=config.ffn_hidden_size,
            dropout=config.dropout,
            bias=config.bias,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with residual connections.
        
        Args:
            x: Input tensor [batch, seq_len, n_embd]
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor [batch, seq_len, n_embd]
        """
        # Attention block with residual
        x = x + self.attn(self.ln1(x), attention_mask)
        
        # FFN block with residual
        x = x + self.ffn(self.ln2(x))
        
        return x


class TransformerLM(nn.Module):
    """
    GPT-style decoder-only transformer language model.
    
    Architecture:
    - Token embeddings
    - Optional positional embeddings (or RoPE in attention)
    - N transformer blocks
    - Layer norm
    - LM head (tied with token embeddings)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Positional embeddings (only if not using RoPE)
        if not config.use_rope:
            self.pos_embedding = nn.Embedding(config.context_length, config.n_embd)
        
        # Dropout
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # LM head (output projection to vocabulary)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights between token embeddings and lm_head
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Calculate and store number of parameters
        self.n_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {self.n_params:,} parameters ({self.n_params / 1e9:.2f}B)")
    
    def _init_weights(self, module):
        """Initialize weights using scaled initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_checkpoint: bool = False,
    ) -> dict:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Optional attention mask [batch, seq_len]
            labels: Optional labels for loss computation [batch, seq_len]
            use_checkpoint: Whether to use gradient checkpointing
            
        Returns:
            Dictionary with logits and optionally loss
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Add positional embeddings if not using RoPE
        if not self.config.use_rope:
            pos_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
            x = x + self.pos_embedding(pos_ids)
        
        x = self.drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            if use_checkpoint and self.training:
                # Use gradient checkpointing to save memory
                x = checkpoint.checkpoint(block, x, attention_mask, use_reentrant=False)
            else:
                x = block(x, attention_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # LM head
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross entropy
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,  # Ignore padding tokens
            )
        
        return {
            'logits': logits,
            'loss': loss,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Starting tokens [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop to context length
            input_ids_cropped = input_ids[:, -self.config.context_length:]
            
            # Forward pass
            outputs = self(input_ids_cropped)
            logits = outputs['logits']
            
            # Get logits for last token
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample from distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Get number of parameters.
        
        Args:
            non_embedding: If True, exclude embedding parameters
            
        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            if not self.config.use_rope:
                n_params -= self.pos_embedding.weight.numel()
        return n_params
