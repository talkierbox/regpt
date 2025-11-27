# regpt

Tiny decoder-only GPT written from scratch with PyTorch. It tokenizes a small classic-literature corpus, trains a stack of multi-head attention blocks, and logs every experiment for quick comparisons.

## Quick start
- `python3 -m venv .venv && source .venv/bin/activate`
- `pip install -r requirements.txt`
- `python -m src.main` (or `./run_main.sh`) to train; artifacts land in `logs/transformer_training_*`.

## Data
Drop any plain-text files into `training_data/`. `src/core/dataset.py` concatenates them, builds a character-level tokenizer, and feeds fixed-length blocks into the transformer.

## Why does this exist?
Mainly as a playground for me to learn more about the architecture from the ground-up.