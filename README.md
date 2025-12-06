# LLM Training Project (~2GB Model)

A from-scratch implementation of a GPT-style transformer language model with ~1.5B parameters (2GB on disk).

## Model Specifications

- **Architecture**: Decoder-only Transformer (GPT-style)
- **Parameters**: ~1.45 billion
- **Layers**: 24
- **Hidden Dimension**: 1536
- **Attention Heads**: 16
- **Context Length**: 2048 tokens
- **Vocabulary**: 50,257 (GPT-2 tokenizer)

## Features

- ✅ Mixed precision training (FP16/BF16)
- ✅ Gradient checkpointing for memory efficiency
- ✅ Flash Attention 2 support
- ✅ Robust checkpoint/resume system
- ✅ Gradient accumulation for large effective batch sizes
- ✅ Automatic interrupt handling (SIGINT/SIGTERM)
- ✅ Weights & Biases / TensorBoard logging
- ✅ Single GPU optimized

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

```bash
# The Pile dataset (~800GB)
python data/preprocessing/download_pile.py
```

### Training

```bash
# Start training from scratch
python scripts/train.py

# Resume from checkpoint (automatic if checkpoint exists)
python scripts/train.py --resume
```

### Configuration

Edit YAML files in `configs/`:
- `model_config.yaml`: Model architecture
- `training_config.yaml`: Training hyperparameters
- `data_config.yaml`: Dataset settings

## Project Structure

```
nn-v4/
├── configs/           # Configuration files
├── models/            # Model architecture
├── data/              # Dataset and tokenizer
├── training/          # Training loop and optimization
├── evaluation/        # Evaluation and generation
├── scripts/           # Main entry points
├── utils/             # Utilities
├── checkpoints/       # Saved checkpoints
└── logs/              # Training logs
```

## Training Details

- **Dataset**: The Pile (300B tokens)
- **Training Tokens**: 300-600B (1-2 epochs)
- **Effective Batch Size**: 128-512 tokens
- **Learning Rate**: 6e-4 with cosine decay
- **Optimizer**: AdamW
- **Expected Duration**: 2-6 months on single GPU (RTX 4090/A100)

## Checkpointing

Checkpoints are saved:
- Every 5,000 steps
- At end of each epoch
- On manual interrupt (Ctrl+C)
- When achieving best validation loss

Training automatically resumes from the latest checkpoint if interrupted.

## GPU Requirements

- **Minimum**: 16GB VRAM (RTX 4080, A4000)
- **Recommended**: 24GB VRAM (RTX 4090, A100 40GB)

With 16GB GPU, use:
- Smaller batch size (1-2)
- 8-bit optimizer
- Consider disabling Flash Attention if compilation issues

## License

MIT License
