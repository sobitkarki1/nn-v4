"""Analyze model quantization and precision."""
import torch
import yaml
from models import ModelConfig, TransformerLM

print("=" * 70)
print("MODEL QUANTIZATION & PRECISION ANALYSIS")
print("=" * 70)

# Load config
with open('configs/model_config.yaml', 'r') as f:
    model_config_dict = yaml.safe_load(f)

# Create model
print("\n1. Creating model...")
model_config = ModelConfig(**model_config_dict['model'])
model = TransformerLM(model_config)

print(f"\n2. Model Architecture:")
print(f"   - Layers: {model_config.n_layer}")
print(f"   - Hidden dim: {model_config.n_embd}")
print(f"   - Attention heads: {model_config.n_head}")
print(f"   - Vocabulary: {model_config.vocab_size:,}")
print(f"   - Context length: {model_config.context_length}")

# Parameter count
total_params = model.n_params
embedding_params = model.token_embedding.weight.numel()
non_embedding_params = model.get_num_params(non_embedding=True)

print(f"\n3. Parameter Count:")
print(f"   - Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
print(f"   - Embedding parameters: {embedding_params:,} ({embedding_params / 1e6:.2f}M)")
print(f"   - Non-embedding parameters: {non_embedding_params:,} ({non_embedding_params / 1e6:.2f}M)")

# Precision analysis
print(f"\n4. DEFAULT PRECISION ANALYSIS:")
print(f"   By default, PyTorch models use float32 (FP32) precision")
print(f"   - Each parameter: 4 bytes")
print(f"   - Model size in FP32: {total_params * 4 / 1e9:.3f} GB")

# Check actual dtype
sample_param = next(model.parameters())
print(f"\n5. Current Model Precision:")
print(f"   - Data type: {sample_param.dtype}")
print(f"   - Bytes per parameter: {sample_param.element_size()}")
print(f"   - Actual model size: {total_params * sample_param.element_size() / 1e9:.3f} GB")

# Different precision options
print(f"\n6. MODEL SIZE AT DIFFERENT PRECISIONS:")
print(f"   - FP32 (float32):  {total_params * 4 / 1e9:.3f} GB  [DEFAULT - No quantization]")
print(f"   - FP16 (float16):  {total_params * 2 / 1e9:.3f} GB  [2x smaller, ~same speed]")
print(f"   - BF16 (bfloat16): {total_params * 2 / 1e9:.3f} GB  [2x smaller, better range than FP16]")
print(f"   - INT8:            {total_params * 1 / 1e9:.3f} GB  [4x smaller, slower inference]")
print(f"   - INT4:            {total_params * 0.5 / 1e9:.3f} GB [8x smaller, quality loss]")

# Convert to different precisions
print(f"\n7. TESTING DIFFERENT PRECISIONS:")

# FP16
model_fp16 = model.half()
print(f"   ✓ FP16 conversion successful")
print(f"     - Data type: {next(model_fp16.parameters()).dtype}")
print(f"     - Size: {total_params * 2 / 1e9:.3f} GB")

# BF16 (if available)
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    model_bf16 = model.bfloat16()
    print(f"   ✓ BF16 conversion successful")
    print(f"     - Data type: {next(model_bf16.parameters()).dtype}")
else:
    print(f"   ⚠ BF16 not supported on this GPU")

# Back to FP32
model = model.float()
print(f"   ✓ Back to FP32")

print(f"\n8. MIXED PRECISION TRAINING:")
print(f"   Our training config uses: mixed_precision = True")
print(f"   This means:")
print(f"   - Forward/backward pass: FP16 (faster, less memory)")
print(f"   - Optimizer state: FP32 (numerical stability)")
print(f"   - Gradients: FP16 (save memory)")
print(f"   - Master weights: FP32 (accumulate updates precisely)")

# Memory estimation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n9. MEMORY REQUIREMENTS (for training):")
batch_size = 1
seq_len = model_config.context_length

print(f"\n   With batch_size={batch_size}, seq_len={seq_len}:")
print(f"   - Model parameters (FP16): {total_params * 2 / 1e9:.3f} GB")
print(f"   - Optimizer states (FP32): {total_params * 8 / 1e9:.3f} GB (2x for Adam momentum)")
print(f"   - Gradients (FP16): {total_params * 2 / 1e9:.3f} GB")
print(f"   - Activations (estimated): ~0.5-1.0 GB (with gradient checkpointing)")
print(f"   - Total estimated: {(total_params * 2 + total_params * 8 + total_params * 2) / 1e9 + 1:.3f} GB")

# Test forward pass
print(f"\n10. TESTING FORWARD PASS:")
model = model.to(device)
input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len), device=device)

with torch.no_grad():
    outputs = model(input_ids)
    print(f"   ✓ Forward pass successful")
    print(f"   - Output shape: {outputs['logits'].shape}")
    print(f"   - Output dtype: {outputs['logits'].dtype}")

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"\n   GPU Memory Usage:")
    print(f"   - Allocated: {allocated:.3f} GB")
    print(f"   - Reserved: {reserved:.3f} GB")

print("\n" + "=" * 70)
print("SUMMARY:")
print("=" * 70)
print(f"✓ Model has {total_params:,} parameters ({total_params / 1e6:.1f}M)")
print(f"✓ Default precision: float32 (FP32) - NO quantization by default")
print(f"✓ Model size: {total_params * 4 / 1e9:.3f} GB in FP32")
print(f"✓ Training uses mixed precision (FP16) automatically")
print(f"✓ Can be quantized to FP16/BF16 for inference (2x smaller)")
print(f"✓ Estimated training memory: {(total_params * 12) / 1e9 + 1:.3f} GB")
print("=" * 70)
