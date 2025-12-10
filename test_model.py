"""Quick test of model initialization and forward pass."""
import torch
import yaml
from models import ModelConfig, TransformerLM

# Load config
with open('configs/model_config.yaml', 'r') as f:
    model_config_dict = yaml.safe_load(f)

# Create model
print("Creating model...")
model_config = ModelConfig(**model_config_dict['model'])
model = TransformerLM(model_config)

print(f"Model created with {model.n_params:,} parameters ({model.n_params / 1e9:.2f}B)")

# Test forward pass
print("\nTesting forward pass...")
batch_size = 1
seq_len = 128

input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len))
labels = torch.randint(0, model_config.vocab_size, (batch_size, seq_len))

with torch.no_grad():
    outputs = model(input_ids, labels=labels)
    
print(f"Input shape: {input_ids.shape}")
print(f"Output logits shape: {outputs['logits'].shape}")
print(f"Loss: {outputs['loss'].item():.4f}")

# Test generation
print("\nTesting generation...")
prompt_ids = torch.randint(0, model_config.vocab_size, (1, 10))
with torch.no_grad():
    generated = model.generate(
        prompt_ids,
        max_new_tokens=20,
        temperature=1.0,
        top_k=50,
    )
print(f"Generated {generated.shape[1]} tokens")

print("\n✅ All tests passed!")
