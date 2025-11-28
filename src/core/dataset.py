import sys
from pathlib import Path
import token

# Add project root to path for both direct execution and module imports
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from torch.utils.data import Dataset
from typing import Optional

from src.core.load_files import fetch_data_files
from src.core.tokenizer import Tokenizer


def fetch_dataset(block_size: int = 128, path_extension: Optional[Path] = None, val_precentage: float = 0.1) -> tuple[Dataset, int]:
    # Returns the dataset as well as the alphabet size

    data_files = fetch_data_files(path_extension)
    all_content: str = "".join(datafile['file_content'] for datafile in data_files.values())
    tokenizer = Tokenizer(all_content)

    train_portion = int(len(all_content) * (1 - val_precentage))

    train_data = all_content[:train_portion]
    val_data = all_content[train_portion:]

    train_dataset = TokenDataset(tokenizer.encode(train_data), block_size)
    val_dataset = TokenDataset(tokenizer.encode(val_data), block_size)

    return train_dataset, val_dataset, len(tokenizer.alphabet)
   

# Dataset for next-token prediction
class TokenDataset(Dataset):

    def __init__(self, tokens: list[int], block_size: int):
        """
        tokens: list[int], of already encoded tokens
        block_size: int, context length for the model
        """

        self.data = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        # All the possible starting positions for a full block
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size] # 1 2 3 4
        y = self.data[idx + 1 : idx + 1 + self.block_size] # 2 3 4 5 

        return x, y


if __name__ == "__main__":
    dataset, _ = fetch_dataset()
    print(len(dataset))