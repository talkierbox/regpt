import sys
from pathlib import Path

from core.model.transformer import DecoderTransformer

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import random
import numpy as np
import tqdm.auto as tqdm

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from src.core.dataset import fetch_dataset
from src.utils import ExperimentLogger

# PARAMS
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
RANDOM_SEED = 16

# Training
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-4

# Data
BLOCK_SIZE = 128

# Model Architecture
D_MODEL = 64
ATTENTION_BLOCKS = 6
NUM_HEADS = 8
DROPOUT = 0.1

# Logging
EXPERIMENT_NAME = "transformer_training"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

torch.device(DEVICE)
dataset, alphabet_size = fetch_dataset(BLOCK_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = DecoderTransformer(
    alphabet_size=alphabet_size,
    d_model=D_MODEL,
    attention_block_count=ATTENTION_BLOCKS,
    num_heads=NUM_HEADS,
    dropout=DROPOUT
)
model = model.to(DEVICE)

# Initialize logger
logger = ExperimentLogger(EXPERIMENT_NAME)
logger.header("Training Configuration")
logger.log(f"Device: {DEVICE}", level="info")
logger.log(f"Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | Block Size: {BLOCK_SIZE}", level="info")
logger.log(f"Learning Rate: {LR} | Alphabet Size: {alphabet_size}", level="info")
logger.log(f"Model: d_model={D_MODEL}, attention_blocks={ATTENTION_BLOCKS}, num_heads={NUM_HEADS}", level="info")

# Export config
config = {
    "device": DEVICE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "block_size": BLOCK_SIZE,
    "learning_rate": LR,
    "alphabet_size": alphabet_size,
    "d_model": D_MODEL,
    "attention_block_count": ATTENTION_BLOCKS,
    "num_heads": NUM_HEADS,
    "dropout": DROPOUT,
    "random_seed": RANDOM_SEED
}
logger.export_json(config, "training_config")

criterion = CrossEntropyLoss()
optim = AdamW(model.parameters(), lr=LR)

logger.header("Starting Training")
epoch_losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    with logger.timer(f"Epoch {epoch + 1}/{EPOCHS}"):
        pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Forward pass
            logits = model(x)  # [B, seq_len, alphabet_size]
            B, seq_len, vocab_size = logits.shape
            
            # Reshape for loss calculation
            logits = logits.reshape(B * seq_len, vocab_size)
            y = y.flatten()
            
            # Compute loss
            loss = criterion(logits, y)
            
            # Backward pass
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            # Track metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update tqdm with current loss
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    # Calculate average epoch loss
    avg_epoch_loss = epoch_loss / num_batches
    epoch_losses.append(avg_epoch_loss)
    
    # Log epoch summary
    logger.log_metrics({
        "epoch_loss": avg_epoch_loss,
    }, step=epoch, prefix="epoch")
    
    logger.log(f"Epoch {epoch + 1} completed - Avg Loss: {avg_epoch_loss:.4f}", level="success")

# Training complete
logger.header("Training Complete")
logger.log(f"Best epoch loss: {min(epoch_losses):.4f} (epoch {epoch_losses.index(min(epoch_losses)) + 1})", level="success")

# Export metrics and charts
logger.export_metrics("training_metrics")
logger.plot_line(
    list(range(1, EPOCHS + 1)),
    epoch_losses,
    "training_loss",
    title="Training Loss Over Epochs",
    xlabel="Epoch",
    ylabel="Loss"
)
logger.plot_metrics()

# Export trained model (use torch.save for PyTorch models)
logger.log("Exporting trained model...", level="info")
model_path = logger.get_path("model_state_dict.pt")
torch.save(model.state_dict(), model_path)
logger.log(f"Model state dict saved to: {model_path}", level="success")

# Generate summary
logger.summarize()


