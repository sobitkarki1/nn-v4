# LLM Training Project (~2GB Model)

## Version History

This is the latest in a personal ML learning series:

| Repo | Params | Key Concept |
|------|--------|-------------|
| [nn-v1](https://github.com/sobitkarki1/nn-v1) | ~1.4K | Backprop from scratch (NumPy) |
| [nn-v2](https://github.com/sobitkarki1/nn-v2) | ~6.8K | PyTorch autograd, bigram LM |
| [nanogpt-legacy](https://github.com/sobitkarki1/nanogpt-legacy) | ~10M | Attention, transformer blocks |
| **nn-v4** (this) | ~1.45B | Full GPT at scale — mixed precision, Flash Attention 2 |

---
 LLM Training Project (~2GB Model)

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

- Mixed precision training (FP16/BF16)
- Gradient checkpointing for memory efficiency
- Flash Attention 2 support
- Robust checkpoint/resume system
- Gradient accumulation for large effective batch sizes
- Automatic interrupt handling (SIGINT/SIGTERM)
- Weights & Biases / TensorBoard logging
- Single GPU optimized

## Quick Start

### Installation

`ash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
`

### Training

`ash
python scripts/train.py          # start from scratch
python scripts/train.py --resume # resume from checkpoint
`

## Training Details

- **Dataset**: The Pile (300B tokens)
- **Effective Batch Size**: 128-512 tokens
- **Learning Rate**: 6e-4 with cosine decay
- **Optimizer**: AdamW
- **Expected Duration**: 2-6 months on single GPU

## GPU Requirements

- **Minimum**: 16GB VRAM (RTX 4080, A4000)
- **Recommended**: 24GB VRAM (RTX 4090, A100 40GB)

## License

MIT License