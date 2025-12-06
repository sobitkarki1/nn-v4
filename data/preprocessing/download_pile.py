"""Download The Pile dataset."""
import os
from datasets import load_dataset


def download_pile(output_dir: str = "data/the_pile", num_proc: int = 4):
    """
    Download The Pile dataset.
    
    Args:
        output_dir: Directory to save dataset
        num_proc: Number of processes for downloading
    """
    print("Downloading The Pile dataset...")
    print("Warning: This is a very large dataset (~800GB)")
    print("Consider using streaming mode instead (enabled by default in config)")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Download dataset
    dataset = load_dataset(
        "EleutherAI/pile",
        split="train",
        cache_dir=output_dir,
        num_proc=num_proc,
    )
    
    print(f"Dataset downloaded to {output_dir}")
    print(f"Number of examples: {len(dataset):,}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download The Pile dataset')
    parser.add_argument('--output_dir', type=str, default='data/the_pile',
                        help='Output directory')
    parser.add_argument('--num_proc', type=int, default=4,
                        help='Number of processes')
    args = parser.parse_args()
    
    download_pile(args.output_dir, args.num_proc)
