"""
Create a tiny local model for fast GRPO smoke testing.
No internet required. Catches all logic bugs (dtype, tokenizer mismatch, 
gradient flow, checkpoint save, log write) without waiting for downloads.
"""
import os
import sys
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CKPT_DIR = "checkpoints/tiny_test_model"

def create_tiny_model():
    os.makedirs(CKPT_DIR, exist_ok=True)
    
    # Tiny GPT-2: 2 layers, 128 hidden, 2 heads
    config = GPT2Config(
        vocab_size=50257,
        n_positions=2048,
        n_embd=128,
        n_layer=2,
        n_head=2,
        n_inner=512,
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(CKPT_DIR)
    
    # GPT2 tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(CKPT_DIR)
    
    print(f"Created tiny model at {CKPT_DIR}")
    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return CKPT_DIR

if __name__ == "__main__":
    create_tiny_model()
