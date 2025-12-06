"""Main training script for LLM."""
import torch
import yaml
import argparse
import signal
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ModelConfig, TransformerLM
from data import Tokenizer, get_dataloader
from training import get_optimizer, get_scheduler, CheckpointManager
from evaluation import evaluate_model
from utils import (
    setup_logging, log_metrics, close_loggers,
    print_gpu_memory, set_seed, MetricsTracker
)


class Trainer:
    """Main trainer class."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configs
        self.load_configs()
        
        # Set random seed
        set_seed(self.train_config['seed'])
        
        # Setup logging
        self.loggers = setup_logging(self.train_config['logging'])
        
        # Initialize model
        self.model_config = ModelConfig(**self.model_config_dict['model'])
        self.model = TransformerLM(self.model_config).to(self.device)
        
        # Compile model if enabled
        if self.train_config['training']['compile_model']:
            try:
                self.model = torch.compile(self.model, mode='max-autotune')
                print("Model compiled with torch.compile")
            except Exception as e:
                print(f"Warning: Could not compile model: {e}")
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer(
            tokenizer_type=self.data_config['dataset']['tokenizer_type'],
            tokenizer_path=self.data_config['dataset'].get('tokenizer_path'),
        )
        
        # Initialize data loaders
        self.train_loader = get_dataloader(
            dataset_name=self.data_config['dataset']['name'],
            dataset_path=self.data_config['dataset']['path'],
            tokenizer=self.tokenizer,
            split=self.data_config['dataset']['train_split'],
            max_length=self.data_config['dataset']['max_length'],
            batch_size=self.train_config['training']['batch_size'],
            num_workers=self.train_config['training']['num_workers'],
            pin_memory=self.train_config['training']['pin_memory'],
            streaming=self.data_config['dataset']['streaming'],
            cache_dir=self.data_config['dataset'].get('cache_dir'),
        )
        
        self.val_loader = get_dataloader(
            dataset_name=self.data_config['dataset']['name'],
            dataset_path=self.data_config['dataset']['path'],
            tokenizer=self.tokenizer,
            split=self.data_config['dataset']['val_split'],
            max_length=self.data_config['dataset']['max_length'],
            batch_size=self.train_config['training']['batch_size'],
            num_workers=self.train_config['training']['num_workers'],
            pin_memory=self.train_config['training']['pin_memory'],
            streaming=self.data_config['dataset']['streaming'],
            cache_dir=self.data_config['dataset'].get('cache_dir'),
        )
        
        # Initialize optimizer
        self.optimizer = get_optimizer(
            self.model,
            learning_rate=self.train_config['training']['learning_rate'],
            weight_decay=self.train_config['training']['weight_decay'],
            beta1=self.train_config['training']['beta1'],
            beta2=self.train_config['training']['beta2'],
            eps=self.train_config['training']['eps'],
            use_8bit=self.train_config['training']['use_8bit_optimizer'],
        )
        
        # Initialize scheduler
        self.scheduler = get_scheduler(
            self.optimizer,
            scheduler_type=self.train_config['training']['lr_scheduler'],
            num_warmup_steps=self.train_config['training']['warmup_steps'],
            num_training_steps=self.train_config['training']['max_steps'],
            min_lr_ratio=self.train_config['training']['min_lr'] / self.train_config['training']['learning_rate'],
        )
        
        # Initialize gradient scaler for mixed precision
        self.scaler = None
        if self.train_config['training']['mixed_precision']:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.train_config['checkpoint']['checkpoint_dir'],
            keep_last_n=self.train_config['checkpoint']['keep_last_n'],
            save_best=self.train_config['checkpoint']['save_best'],
        )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.tokens_seen = 0
        
        # Metrics tracker
        self.metrics_tracker = MetricsTracker(window_size=100)
        
        # Setup interrupt handler
        if self.train_config['checkpoint']['save_on_interrupt']:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
    
    def load_configs(self):
        """Load configuration files."""
        with open('configs/model_config.yaml', 'r') as f:
            self.model_config_dict = yaml.safe_load(f)
        
        with open('configs/training_config.yaml', 'r') as f:
            self.train_config = yaml.safe_load(f)
        
        with open('configs/data_config.yaml', 'r') as f:
            self.data_config = yaml.safe_load(f)
    
    def _signal_handler(self, sig, frame):
        """Handle interrupt signals."""
        print('\n\nReceived interrupt signal. Saving checkpoint...')
        self.save_checkpoint(is_interrupt=True)
        close_loggers(self.loggers)
        sys.exit(0)
    
    def save_checkpoint(self, val_loss=None, is_interrupt=False):
        """Save training checkpoint."""
        filename = "checkpoint_interrupt.pt" if is_interrupt else None
        
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=self.epoch,
            global_step=self.global_step,
            tokens_seen=self.tokens_seen,
            train_loss=self.metrics_tracker.get_average_loss(),
            val_loss=val_loss,
            model_config=self.model_config_dict,
            filename=filename,
        )
    
    def resume_from_checkpoint(self):
        """Resume training from latest checkpoint."""
        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
        
        if latest_checkpoint:
            checkpoint_info = self.checkpoint_manager.load_checkpoint(
                filepath=latest_checkpoint,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                device=self.device,
            )
            
            self.epoch = checkpoint_info['epoch']
            self.global_step = checkpoint_info['global_step']
            self.tokens_seen = checkpoint_info['tokens_seen']
            
            print(f"Resumed training from step {self.global_step}")
            return True
        else:
            print("No checkpoint found, starting from scratch")
            return False
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate model on validation set."""
        print("\nEvaluating...")
        eval_results = evaluate_model(
            self.model,
            self.val_loader,
            max_batches=self.train_config['logging']['eval_batches'],
            device=self.device,
        )
        
        print(f"Validation Loss: {eval_results['loss']:.4f}, "
              f"Perplexity: {eval_results['perplexity']:.2f}")
        
        return eval_results
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training on {self.device}")
        print(f"Model size: {self.model.n_params:,} parameters")
        print_gpu_memory()
        
        # Resume from checkpoint if exists
        if self.args.resume or self.checkpoint_manager.get_latest_checkpoint():
            self.resume_from_checkpoint()
        
        self.model.train()
        
        # Training parameters
        max_steps = self.train_config['training']['max_steps']
        grad_accum_steps = self.train_config['training']['gradient_accumulation_steps']
        max_grad_norm = self.train_config['training']['max_grad_norm']
        log_interval = self.train_config['logging']['log_interval']
        eval_interval = self.train_config['logging']['eval_interval']
        save_interval = self.train_config['checkpoint']['save_interval']
        use_amp = self.train_config['training']['mixed_precision']
        use_grad_checkpoint = self.train_config['training']['gradient_checkpointing']
        
        # Training loop
        pbar = tqdm(total=max_steps, initial=self.global_step, desc="Training")
        
        while self.global_step < max_steps:
            for batch in self.train_loader:
                if self.global_step >= max_steps:
                    break
                
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = self.model(
                        input_ids,
                        labels=labels,
                        use_checkpoint=use_grad_checkpoint,
                    )
                    loss = outputs['loss'] / grad_accum_steps
                
                # Backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights every grad_accum_steps
                if (self.global_step + 1) % grad_accum_steps == 0:
                    # Gradient clipping
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    
                    # Optimizer step
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                # Update metrics
                num_tokens = (labels != -100).sum().item()
                self.tokens_seen += num_tokens
                self.metrics_tracker.update(
                    loss=loss.item() * grad_accum_steps,
                    learning_rate=self.scheduler.get_last_lr()[0],
                    num_tokens=num_tokens,
                )
                
                self.global_step += 1
                pbar.update(1)
                
                # Logging
                if self.global_step % log_interval == 0:
                    metrics = {
                        'train/loss': self.metrics_tracker.get_average_loss(),
                        'train/learning_rate': self.metrics_tracker.get_current_lr(),
                        'train/tokens_seen': self.tokens_seen,
                        'train/epoch': self.epoch,
                    }
                    
                    # Add GPU memory
                    if torch.cuda.is_available():
                        from utils.gpu_utils import get_gpu_memory_info
                        gpu_info = get_gpu_memory_info()
                        metrics['train/gpu_memory_gb'] = gpu_info['allocated_gb']
                    
                    log_metrics(self.loggers, metrics, self.global_step)
                    
                    pbar.set_postfix({
                        'loss': f"{metrics['train/loss']:.4f}",
                        'lr': f"{metrics['train/learning_rate']:.2e}",
                    })
                
                # Evaluation
                if self.global_step % eval_interval == 0:
                    eval_results = self.evaluate()
                    
                    log_metrics(self.loggers, {
                        'val/loss': eval_results['loss'],
                        'val/perplexity': eval_results['perplexity'],
                    }, self.global_step)
                    
                    self.model.train()
                    
                    # Save if best
                    self.save_checkpoint(val_loss=eval_results['loss'])
                
                # Save checkpoint
                if self.global_step % save_interval == 0:
                    self.save_checkpoint()
            
            self.epoch += 1
        
        pbar.close()
        
        # Final save
        print("\nTraining complete! Saving final checkpoint...")
        self.save_checkpoint()
        
        # Close loggers
        close_loggers(self.loggers)
        
        print(f"\nFinal stats:")
        print(f"Total steps: {self.global_step:,}")
        print(f"Total tokens: {self.tokens_seen:,}")
        print_gpu_memory()


def main():
    parser = argparse.ArgumentParser(description='Train LLM')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()
    
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
