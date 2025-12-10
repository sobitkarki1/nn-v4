"""Minimal training test script."""
import torch
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ModelConfig, TransformerLM

print("Loading configurations...")
with open('configs/model_config.yaml', 'r') as f:
    model_config_dict = yaml.safe_load(f)

with open('configs/training_config.yaml', 'r') as f:
    train_config = yaml.safe_load(f)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create model
print("\nInitializing model...")
model_config = ModelConfig(**model_config_dict['model'])
model = TransformerLM(model_config).to(device)

print(f"Model: {model.n_params:,} parameters ({model.n_params / 1e9:.2f}B)")

# Test training configuration
batch_size = train_config['training']['batch_size']
max_steps = 100  # Just 100 steps for testing
learning_rate = train_config['training']['learning_rate']

print(f"\nTraining configuration:")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {learning_rate}")
print(f"  Mixed precision: {train_config['training']['mixed_precision']}")
print(f"  Gradient checkpointing: {train_config['training']['gradient_checkpointing']}")

# Create dummy data
print(f"\nCreating dummy training data...")
seq_len = model_config.context_length

# Simple training loop
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=learning_rate)

print(f"\nStarting training for {max_steps} steps...")
model.train()

for step in range(max_steps):
    # Generate random data (in real training, this comes from dataloader)
    input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, model_config.vocab_size, (batch_size, seq_len), device=device)
    
    # Forward pass
    outputs = model(input_ids, labels=labels)
    loss = outputs['loss']
    
    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Log every 10 steps
    if (step + 1) % 10 == 0:
        print(f"Step {step + 1}/{max_steps} - Loss: {loss.item():.4f}")

print(f"\n✅ Training test completed successfully!")
print(f"Final loss: {loss.item():.4f}")

# Test saving
print("\nTesting checkpoint save...")
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'step': max_steps,
}
torch.save(checkpoint, 'checkpoints/test_checkpoint.pt')
print("✅ Checkpoint saved to checkpoints/test_checkpoint.pt")

# Test loading
print("\nTesting checkpoint load...")
loaded = torch.load('checkpoints/test_checkpoint.pt', map_location=device)
model.load_state_dict(loaded['model_state_dict'])
print("✅ Checkpoint loaded successfully")

print("\n🎉 All tests passed! Training system is working correctly.")
