"""Training metrics tracking."""
from collections import deque
from typing import Optional


class MetricsTracker:
    """
    Track training metrics with moving averages.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Window size for moving average
        """
        self.window_size = window_size
        self.losses = deque(maxlen=window_size)
        self.learning_rates = deque(maxlen=window_size)
        
        self.total_tokens = 0
        self.total_steps = 0
    
    def update(
        self,
        loss: float,
        learning_rate: float,
        num_tokens: int,
    ):
        """
        Update metrics.
        
        Args:
            loss: Current loss
            learning_rate: Current learning rate
            num_tokens: Number of tokens in batch
        """
        self.losses.append(loss)
        self.learning_rates.append(learning_rate)
        self.total_tokens += num_tokens
        self.total_steps += 1
    
    def get_average_loss(self) -> float:
        """Get average loss over window."""
        if not self.losses:
            return 0.0
        return sum(self.losses) / len(self.losses)
    
    def get_current_lr(self) -> float:
        """Get current learning rate."""
        if not self.learning_rates:
            return 0.0
        return self.learning_rates[-1]
    
    def get_metrics(self) -> dict:
        """Get all metrics as dictionary."""
        return {
            'loss': self.get_average_loss(),
            'learning_rate': self.get_current_lr(),
            'total_tokens': self.total_tokens,
            'total_steps': self.total_steps,
        }
