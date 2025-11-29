#!/usr/bin/env python3
"""
Utility script to load the newest trained transformer checkpoint from logs/
and run simple autoregressive inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

# Ensure project root is on sys.path so we can import src.* modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.model.transformer import DecoderTransformer
from src.core.tokenizer import Tokenizer
from src.core.load_files import fetch_data_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the latest trained checkpoint.")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Seed text for generation.")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Number of tokens to sample.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=None, help="Optional top-k filtering.")
    parser.add_argument("--logs-root", type=str, default=str(PROJECT_ROOT / "logs"), help="Directory that holds experiment folders.")
    parser.add_argument("--data-dir", type=str, default="training_data/", help="Directory used to build the tokenizer (same as training).")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu|cuda|mps). Defaults to best available.")
    return parser.parse_args()


def resolve_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available(): 
        return torch.device("mps")
    return torch.device("cpu")


def find_latest_checkpoint(logs_root: Path) -> Tuple[Path, Path]:
    """Return (checkpoint_path, config_path) for the newest experiment that has both."""
    if not logs_root.exists():
        raise FileNotFoundError(f"Logs root '{logs_root}' does not exist.")

    candidate_dirs = sorted(
        [p for p in logs_root.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for exp_dir in candidate_dirs:
        ckpt = exp_dir / "model_state_dict.pt"
        config = exp_dir / "training_config.json"
        if ckpt.exists() and config.exists():
            return ckpt, config

    raise FileNotFoundError("Could not find any checkpoint (.pt) inside logs/. Did training finish?")


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def build_tokenizer(data_dir: str | Path) -> Tokenizer:
    files = fetch_data_files(data_dir)
    if not files:
        raise RuntimeError(f"No files found under '{data_dir}'.")
    corpus = "".join(file["file_content"] for file in files.values())
    if not corpus:
        raise RuntimeError(f"Files under '{data_dir}' are empty, cannot build tokenizer.")
    return Tokenizer(corpus)


def top_k_filter(logits: torch.Tensor, k: Optional[int]) -> torch.Tensor:
    if k is None or k <= 0:
        return logits
    k = min(k, logits.size(-1))
    values, _ = torch.topk(logits, k)
    threshold = values[:, -1].unsqueeze(-1)
    filtered = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)
    return filtered


@torch.no_grad()
def generate_text(
    model: DecoderTransformer,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    *,
    block_size: int,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
) -> str:
    if not prompt:
        raise ValueError("Prompt text must be non-empty.")

    alphabet = set(tokenizer.alphabet)
    unknown_chars = sorted({c for c in prompt if c not in alphabet})
    if unknown_chars:
        raise ValueError(f"Prompt contains characters not in the tokenizer alphabet: {unknown_chars}")

    token_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        context = token_ids[:, -block_size:]
        logits = model(context)
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        logits = top_k_filter(logits, top_k)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_ids = torch.cat([token_ids, next_token], dim=1)

    return tokenizer.decode(token_ids[0].tolist())


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    logs_root = Path(args.logs_root)
    checkpoint_path, config_path = find_latest_checkpoint(logs_root)
    config = load_config(config_path)

    tokenizer = build_tokenizer(args.data_dir)
    if len(tokenizer.alphabet) != config.get("alphabet_size"):
        print(
            f"[warning] Tokenizer alphabet ({len(tokenizer.alphabet)}) differs from config ({config.get('alphabet_size')}). "
            "Generation may not exactly match training."
        )

    model = DecoderTransformer(
        alphabet_size=len(tokenizer.alphabet),
        d_model=config["d_model"],
        attention_block_count=config["attention_block_count"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Using device: {device}")

    output = generate_text(
        model,
        tokenizer,
        args.prompt,
        device,
        block_size=config["block_size"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("\n=== Generated Text ===")
    print(output)


if __name__ == "__main__":
    main()

