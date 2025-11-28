import sys
from pathlib import Path

from core.model.transformer import DecoderTransformer
# TODO: Validation split

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import math
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
EPOCHS = 10
BATCH_SIZE = 32
LR = 3e-4

# Data
BLOCK_SIZE = 128


def _reshape_for_loss(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten logits/targets so CrossEntropyLoss operates over vocab dimension.
    """
    B, seq_len, vocab_size = logits.shape
    return logits.reshape(B * seq_len, vocab_size), targets.reshape(B * seq_len)


def train_one_epoch(
    model: DecoderTransformer,
    dataloader: DataLoader,
    criterion: CrossEntropyLoss,
    optimizer: AdamW,
    device: torch.device,
    epoch_idx: int,
    total_epochs: int,
    logger: ExperimentLogger,
) -> float:
    model.train()
    running_loss = 0.0
    batch_count = 0

    progress = tqdm.tqdm(dataloader, desc=f"Train {epoch_idx}/{total_epochs}", leave=False)
    for x, y in progress:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        logits, targets = _reshape_for_loss(logits, y)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        batch_count += 1
        progress.set_postfix(loss=f"{loss.item():.4f}", ppl=f"{math.exp(loss.item()):.2f}")

    if batch_count == 0:
        raise RuntimeError(
            "Train dataloader produced zero batches. Add more data or reduce BLOCK_SIZE."
        )

    return running_loss / batch_count


@torch.no_grad()
def validate_one_epoch(
    model: DecoderTransformer,
    dataloader: DataLoader,
    criterion: CrossEntropyLoss,
    device: torch.device,
    epoch_idx: int,
    total_epochs: int,
    logger: ExperimentLogger,
) -> float:
    model.eval()
    running_loss = 0.0
    batch_count = 0

    progress = tqdm.tqdm(dataloader, desc=f"Val {epoch_idx}/{total_epochs}", leave=False)
    for x, y in progress:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        logits, targets = _reshape_for_loss(logits, y)
        loss = criterion(logits, targets)

        running_loss += loss.item()
        batch_count += 1
        progress.set_postfix(loss=f"{loss.item():.4f}", ppl=f"{math.exp(loss.item()):.2f}")

    if batch_count == 0:
        logger.warning(
            "Validation dataloader produced zero batches. Validation metrics will be NaN."
        )
        return float("nan")

    return running_loss / batch_count


# Model Architecture
D_MODEL = 32
ATTENTION_BLOCKS = 6
NUM_HEADS = 4
DROPOUT = 0.15

# Logging
EXPERIMENT_NAME = "transformer_training"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

torch.device(DEVICE)
train_dataset, val_dataset, alphabet_size = fetch_dataset(BLOCK_SIZE)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

# Model stats
num_params = sum(p.numel() for p in model.parameters())
num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
random_baseline = math.log(alphabet_size)
logger.log(f"Model Parameters: {num_params:,} total ({num_trainable:,} trainable)", level="info")
logger.log(f"Random baseline loss: {random_baseline:.2f} (perplexity: {alphabet_size})", level="info")

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
    "random_seed": RANDOM_SEED,
    "num_parameters": num_params,
    "random_baseline_loss": random_baseline,
}
logger.export_json(config, "training_config")

criterion = CrossEntropyLoss()
optim = AdamW(model.parameters(), lr=LR)

logger.header("Starting Training")
train_losses = []
val_losses = []

for epoch in range(1, EPOCHS + 1):
    with logger.timer(f"Epoch {epoch}/{EPOCHS}"):
        train_loss = train_one_epoch(
            model,
            train_dataloader,
            criterion,
            optim,
            DEVICE,
            epoch,
            EPOCHS,
            logger,
        )

    val_loss = validate_one_epoch(
        model,
        val_dataloader,
        criterion,
        DEVICE,
        epoch,
        EPOCHS,
        logger,
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_ppl = math.exp(train_loss)
    val_ppl = math.exp(val_loss) if math.isfinite(val_loss) else float("nan")

    logger.log_metrics(
        {
            "loss": train_loss,
            "perplexity": train_ppl,
        },
        step=epoch,
        prefix="train",
    )

    logger.log_metrics(
        {
            "loss": val_loss,
            "perplexity": val_ppl,
        },
        step=epoch,
        prefix="val",
    )

    logger.log(
        f"Epoch {epoch}/{EPOCHS} "
        f"- Train Loss: {train_loss:.4f} (ppl {train_ppl:.2f}) "
        f"| Val Loss: {val_loss:.4f} (ppl {val_ppl:.2f})",
        level="success",
    )

# Training complete
logger.header("Training Complete")
finite_val_losses = [loss for loss in val_losses if math.isfinite(loss)]
if finite_val_losses:
    best_loss = min(finite_val_losses)
    best_epoch = val_losses.index(best_loss) + 1
    best_split = "validation"
else:
    best_loss = min(train_losses)
    best_epoch = train_losses.index(best_loss) + 1
    best_split = "training"
best_ppl = math.exp(best_loss)
logger.log(
    f"Best {best_split} epoch: {best_epoch} - Loss: {best_loss:.4f} | Perplexity: {best_ppl:.2f}",
    level="success",
)

# Export metrics and charts
logger.export_metrics("training_metrics")
epoch_axis = list(range(1, EPOCHS + 1))
plot_data = {"Train Loss": (epoch_axis, train_losses)}

if any(math.isfinite(loss) for loss in val_losses):
    plot_data["Val Loss"] = (epoch_axis, val_losses)

if len(plot_data) == 1:
    logger.plot_line(
        epoch_axis,
        train_losses,
        "training_loss",
        title="Training Loss Over Epochs",
        xlabel="Epoch",
        ylabel="Loss"
    )
else:
    logger.plot_lines(
        plot_data,
        "training_loss",
        title="Training vs Validation Loss",
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


