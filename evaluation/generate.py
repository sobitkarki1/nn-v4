"""Text generation utilities."""
import torch
from typing import Optional, List
from ..models import TransformerLM
from ..data import Tokenizer


@torch.no_grad()
def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str = "",
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
    device: str = "cuda",
) -> str:
    """
    Generate text from a prompt.
    
    Args:
        model: Trained language model
        tokenizer: Tokenizer instance
        prompt: Input prompt text
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling (None to disable)
        top_p: Nucleus sampling (None to disable)
        device: Device to run generation on
        
    Returns:
        Generated text
    """
    model.eval()
    
    # Encode prompt
    if prompt:
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    else:
        # Start with BOS token
        input_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)
    
    # Generate
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0])
    
    return generated_text


def batch_generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompts: List[str],
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
    device: str = "cuda",
) -> List[str]:
    """
    Generate text for multiple prompts.
    
    Args:
        model: Trained language model
        tokenizer: Tokenizer instance
        prompts: List of input prompts
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling
        device: Device to run generation on
        
    Returns:
        List of generated texts
    """
    model.eval()
    
    generated_texts = []
    for prompt in prompts:
        text = generate_text(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
        )
        generated_texts.append(text)
    
    return generated_texts


def interactive_generation(
    model: TransformerLM,
    tokenizer: Tokenizer,
    device: str = "cuda",
):
    """
    Interactive text generation loop.
    
    Args:
        model: Trained language model
        tokenizer: Tokenizer instance
        device: Device to run generation on
    """
    print("Interactive generation mode. Type 'quit' to exit.")
    print("Parameters: max_tokens=100, temperature=0.8, top_p=0.95")
    print("-" * 50)
    
    model.eval()
    
    while True:
        prompt = input("\nPrompt: ")
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        # Generate
        generated = generate_text(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.8,
            top_p=0.95,
            device=device,
        )
        
        print(f"\nGenerated:\n{generated}")
        print("-" * 50)
