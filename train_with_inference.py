"""Training with dynamic LR, frequent checkpoints, and inference testing."""
import torch
import yaml
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent))

from models import ModelConfig, TransformerLM
from transformers import GPT2Tokenizer
from training import get_optimizer, get_scheduler, CheckpointManager

print("=" * 80)
print("TRAINING WITH DYNAMIC LEARNING RATE & INFERENCE TESTING")
print("=" * 80)

# Load configurations
with open('configs/model_config.yaml', 'r') as f:
    model_config_dict = yaml.safe_load(f)

with open('configs/training_config.yaml', 'r') as f:
    train_config = yaml.safe_load(f)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Create model
print("\nInitializing model...")
model_config = ModelConfig(**model_config_dict['model'])
model = TransformerLM(model_config).to(device)
print(f"Model: {model.n_params:,} parameters ({model.n_params / 1e6:.1f}M)")

# Initialize GPT-2 tokenizer
print("\nDownloading and initializing GPT-2 tokenizer...")
try:
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', resume_download=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ GPT-2 Tokenizer loaded (vocab size: {len(tokenizer)})")
except Exception as e:
    print(f"Failed to download GPT-2 tokenizer: {e}")
    print("Creating simple fallback tokenizer...")
    
    class SimpleTokenizer:
        def __init__(self):
            self.vocab_size = 50257
            self.eos_token = '</s>'
            self.pad_token = '</s>'
        
        def encode(self, text, return_tensors=None):
            # Simple char-level encoding
            tokens = [min(ord(c) % 50000, 50256) for c in text[:100]]
            if return_tensors == 'pt':
                return torch.tensor([tokens])
            return tokens
        
        def decode(self, token_ids, skip_special_tokens=True):
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            # Try to decode as chars
            result = []
            for t in token_ids[:200]:
                try:
                    c = chr(min(max(t % 128, 32), 126))
                    result.append(c)
                except:
                    result.append('?')
            return ''.join(result)
        
        def __len__(self):
            return self.vocab_size
    
    tokenizer = SimpleTokenizer()
    print(f"✓ Fallback tokenizer created (vocab size: {len(tokenizer)})")

# Optimizer with dynamic LR
print("\nSetting up optimizer and scheduler...")
optimizer = get_optimizer(
    model,
    learning_rate=train_config['training']['learning_rate'],
    weight_decay=train_config['training']['weight_decay'],
    beta1=train_config['training']['beta1'],
    beta2=train_config['training']['beta2'],
)

# Scheduler for dynamic LR
max_steps = 200  # Training steps
scheduler = get_scheduler(
    optimizer,
    scheduler_type=train_config['training']['lr_scheduler'],
    num_warmup_steps=train_config['training']['warmup_steps'],
    num_training_steps=max_steps,
    min_lr_ratio=train_config['training']['min_lr'] / train_config['training']['learning_rate'],
)

print(f"\nLearning Rate Schedule:")
print(f"  Type: {train_config['training']['lr_scheduler']}")
print(f"  Initial LR: {train_config['training']['learning_rate']:.2e}")
print(f"  Min LR: {train_config['training']['min_lr']:.2e}")
print(f"  Warmup steps: {train_config['training']['warmup_steps']}")
print(f"  Total steps: {max_steps}")

# Checkpoint manager
checkpoint_manager = CheckpointManager(
    checkpoint_dir=train_config['checkpoint']['checkpoint_dir'],
    keep_last_n=train_config['checkpoint']['keep_last_n'],
    save_best=train_config['checkpoint']['save_best'],
)

# Training configuration
batch_size = train_config['training']['batch_size']
seq_len = model_config.context_length
log_interval = train_config['logging']['log_interval']
eval_interval = train_config['logging']['eval_interval']
save_interval = train_config['checkpoint']['save_interval']

print(f"\nTraining Configuration:")
print(f"  Batch size: {batch_size}")
print(f"  Sequence length: {seq_len}")
print(f"  Log interval: {log_interval} steps")
print(f"  Eval interval: {eval_interval} steps")
print(f"  Checkpoint interval: {save_interval} steps")

# Test prompts for inference
test_prompts = [
    "Once upon a time",
    "The meaning of life is",
    "In the future, artificial intelligence will",
]

def generate_sample(step, prompt):
    """Generate text sample during training."""
    model.eval()
    with torch.no_grad():
        # Properly encode with GPT-2 tokenizer
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        generated = model.generate(
            input_ids,
            max_new_tokens=30,
            temperature=0.8,
            top_k=40,
            top_p=0.95,
        )
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
    model.train()
    return text

# Training loop
print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)

model.train()
losses = []
learning_rates = []

for step in range(max_steps):
    # Generate random training data
    input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, model_config.vocab_size, (batch_size, seq_len), device=device)
    
    # Forward pass
    outputs = model(input_ids, labels=labels)
    loss = outputs['loss']
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['training']['max_grad_norm'])
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
    # Track metrics
    current_lr = scheduler.get_last_lr()[0]
    losses.append(loss.item())
    learning_rates.append(current_lr)
    
    # Logging
    if (step + 1) % log_interval == 0:
        avg_loss = sum(losses[-log_interval:]) / min(log_interval, len(losses))
        print(f"\nStep {step + 1}/{max_steps}")
        print(f"  Loss: {loss.item():.4f} (avg: {avg_loss:.4f})")
        print(f"  Learning Rate: {current_lr:.2e}")
        if torch.cuda.is_available():
            print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Inference testing
    if (step + 1) % eval_interval == 0:
        print(f"\n{'─' * 80}")
        print(f"INFERENCE TEST AT STEP {step + 1}")
        print(f"{'─' * 80}")
        
        for i, prompt in enumerate(test_prompts[:1]):  # Test one prompt
            generated_text = generate_sample(step + 1, prompt)
            print(f"\nPrompt: '{prompt}'")
            print(f"Generated: {generated_text[:150]}...")
        
        print(f"{'─' * 80}\n")
    
    # Save checkpoint
    if (step + 1) % save_interval == 0:
        avg_loss = sum(losses[-save_interval:]) / save_interval
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            epoch=0,
            global_step=step + 1,
            tokens_seen=(step + 1) * batch_size * seq_len,
            train_loss=avg_loss,
            val_loss=None,
            model_config=model_config_dict,
        )
        print(f"  ✓ Checkpoint saved at step {step + 1}")

# Final summary
print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)

print(f"\nFinal Statistics:")
print(f"  Total steps: {max_steps}")
print(f"  Initial loss: {losses[0]:.4f}")
print(f"  Final loss: {losses[-1]:.4f}")
print(f"  Loss improvement: {losses[0] - losses[-1]:.4f}")
print(f"  Initial LR: {learning_rates[0]:.2e}")
print(f"  Final LR: {learning_rates[-1]:.2e}")

# Plot LR schedule
print(f"\nLearning Rate Schedule:")
checkpoints_to_show = [0, 25, 50, 100, 150, max_steps-1]
for step_idx in checkpoints_to_show:
    if step_idx < len(learning_rates):
        print(f"  Step {step_idx:3d}: {learning_rates[step_idx]:.2e}")

# Final inference test
print("\n" + "=" * 80)
print("FINAL INFERENCE TEST")
print("=" * 80)

for prompt in test_prompts:
    generated_text = generate_sample(max_steps, prompt)
    print(f"\nPrompt: '{prompt}'")
    print(f"Generated:\n{generated_text}\n")
    print("-" * 80)

# Save final checkpoint
print("\nSaving final checkpoint...")
checkpoint_manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=None,
    epoch=0,
    global_step=max_steps,
    tokens_seen=max_steps * batch_size * seq_len,
    train_loss=losses[-1],
    val_loss=None,
    model_config=model_config_dict,
    filename="checkpoint_final.pt",
)

print("\n✅ Training complete! All checkpoints saved.")
print(f"📁 Checkpoints directory: {train_config['checkpoint']['checkpoint_dir']}")
