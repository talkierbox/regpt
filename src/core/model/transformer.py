import torch
import numpy as np

from torch import float32, nn, functional, softmax
from src.core.model.mha import MultiheadSelfAttention
from src.core.model.feedforward import FeedForward

class DecoderTransformer(nn.Module):

    def __init__(self, alphabet_size: int, d_model: int = 64, attention_block_count: int = 6, num_heads: int = 8, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(alphabet_size, d_model)

        self.mha_blocks = nn.ModuleList([MultiheadSelfAttention(num_heads=num_heads, d_model=d_model, dropout=dropout) for _ in range(attention_block_count)])
        self.ff_blocks = nn.ModuleList([FeedForward([d_model, 2 * d_model, 2 * d_model, d_model], dropout=dropout) for _ in range(attention_block_count)])

        self.projection = nn.Linear(d_model, alphabet_size)
        
        self.ln1_blocks = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(attention_block_count)])
        self.ln2_blocks = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(attention_block_count)])

    def forward(self, x: torch.Tensor):
        B, seq_len = x.shape

        x = self.tok_emb(x) 
        x = x + self._get_pos_embeddings(B, seq_len, x.device) # [B seq_len d_model]

        # Run MHA blocks
        cur_z = x
        for mha_block, ff, ln1, ln2 in zip(self.mha_blocks, self.ff_blocks, self.ln1_blocks, self.ln2_blocks):
            cur_z = ln1(mha_block(cur_z) + cur_z)
            cur_z = ln2(ff(cur_z) + cur_z)

        # Project
        cur_z = self.projection(cur_z) # [B seq_len alphabet_size]
        
        # Do not run the last softmax. Only run the softmax on inference when generating tokens
        return cur_z

    # TODO: Use a buffer for these in the future so that we don't have to recompute these every single time
    def _get_pos_embeddings(self, B: int, seq_len: int, device: torch.device) -> torch.Tensor:
        # Returns a tensor of positional embeddings of shape [B seq_len d_model]
        return torch.stack([
            self._get_embedding_for_row(i) for i in range(seq_len)
        ]).float().to(device).unsqueeze(0).expand(B, -1, -1)

    def _get_embedding_for_row(self, pos: int) -> torch.Tensor:
        # Generate one row embedding for row_idx of length d_model
        ans = []

        for i in range(self.d_model):
            if i % 2 == 0:
                ans.append(np.sin(pos / (10000 ** (i / self.d_model)) ))
            else:
                ans.append(np.cos(pos / (10000 ** ((i - 0.5) / self.d_model))))
        
        return torch.tensor(ans, dtype=float32)