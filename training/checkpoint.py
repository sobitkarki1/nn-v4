"""Checkpoint save/load utilities."""
import torch
import os
import glob
from typing import Optional, Dict, Any
from datetime import datetime
import random
import numpy as np


class CheckpointManager:
    """
    Manages checkpoint saving, loading, and cleanup.
    
    Features:
    - Automatic checkpoint saving at intervals
    - Keep only last N checkpoints to save disk space
    - Save best checkpoint based on validation loss
    - Full state restoration (model, optimizer, scheduler, RNG states)
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        keep_last_n: int = 5,
        save_best: bool = True,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep
            save_best: Whether to save best checkpoint
        """
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.best_val_loss = float('inf')
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Optional[torch.cuda.amp.GradScaler],
        epoch: int,
        global_step: int,
        tokens_seen: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        model_config: Optional[Dict] = None,
        is_best: bool = False,
        filename: Optional[str] = None,
    ):
        """
        Save checkpoint with full training state.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Learning rate scheduler to save
            scaler: Gradient scaler for mixed precision
            epoch: Current epoch
            global_step: Current global step
            tokens_seen: Total tokens seen
            train_loss: Current training loss
            val_loss: Current validation loss
            model_config: Model configuration
            is_best: Whether this is the best checkpoint
            filename: Custom filename (optional)
        """
        checkpoint = {
            # Model
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            
            # Training state
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            
            # Progress tracking
            'epoch': epoch,
            'global_step': global_step,
            'tokens_seen': tokens_seen,
            
            # Metrics
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            
            # Reproducibility
            'rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
            
            # Metadata
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
        }
        
        # Add gradient scaler state if using mixed precision
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        # Add CUDA RNG state if available
        if torch.cuda.is_available():
            checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state()
        
        # Determine filename
        if filename is None:
            if is_best:
                filename = "checkpoint_best.pt"
            else:
                filename = f"checkpoint_step_{global_step}.pt"
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint to {filepath}")
        
        # Update best checkpoint if needed
        if self.save_best and val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_filepath = os.path.join(self.checkpoint_dir, "checkpoint_best.pt")
            torch.save(checkpoint, best_filepath)
            print(f"New best checkpoint! Val loss: {val_loss:.4f}")
        
        # Also save as latest
        latest_filepath = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
        torch.save(checkpoint, latest_filepath)
        
        # Cleanup old checkpoints
        if not is_best and filename.startswith("checkpoint_step_"):
            self._cleanup_old_checkpoints()
    
    def load_checkpoint(
        self,
        filepath: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore training state.
        
        Args:
            filepath: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            scaler: Gradient scaler to load state into (optional)
            device: Device to load checkpoint to
            
        Returns:
            Dictionary with checkpoint metadata
        """
        print(f"Loading checkpoint from {filepath}")
        checkpoint = torch.load(filepath, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Restore RNG states for reproducibility
        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'].cpu())
        if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu())
        if 'numpy_rng_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_rng_state'])
        if 'python_rng_state' in checkpoint:
            random.setstate(checkpoint['python_rng_state'])
        
        # Update best val loss
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Resumed from epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")
        print(f"Tokens seen: {checkpoint['tokens_seen']:,}")
        
        return {
            'epoch': checkpoint['epoch'],
            'global_step': checkpoint['global_step'],
            'tokens_seen': checkpoint['tokens_seen'],
            'train_loss': checkpoint.get('train_loss', 0.0),
            'val_loss': checkpoint.get('val_loss', None),
        }
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        # Get all step checkpoints
        checkpoint_pattern = os.path.join(self.checkpoint_dir, "checkpoint_step_*.pt")
        checkpoints = sorted(glob.glob(checkpoint_pattern))
        
        # Remove oldest checkpoints
        if len(checkpoints) > self.keep_last_n:
            for ckpt in checkpoints[:-self.keep_last_n]:
                os.remove(ckpt)
                print(f"Removed old checkpoint: {ckpt}")
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get path to latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None if no checkpoint exists
        """
        latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
        if os.path.exists(latest_path):
            return latest_path
        
        # Fall back to most recent step checkpoint
        checkpoint_pattern = os.path.join(self.checkpoint_dir, "checkpoint_step_*.pt")
        checkpoints = sorted(glob.glob(checkpoint_pattern))
        if checkpoints:
            return checkpoints[-1]
        
        return None
