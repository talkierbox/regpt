import torch
from torch import nn, softmax

class MultiheadSelfAttention(nn.Module):
    def __init__(self, num_heads: int = 8, d_model: int = 64, mask: bool = False, dropout=0.1):
        assert d_model % num_heads == 0, f"d_model ({d_model}) is not divisible by num_heads ({num_heads})"
        super().__init__()
        self.d_model = d_model

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.use_mask = mask
        
        self.dropout = nn.Dropout(dropout)

        self.W_q, self.W_k, self.W_v = nn.Linear(self.d_model, self.d_model), nn.Linear(self.d_model, self.d_model), nn.Linear(self.d_model, self.d_model) 
        self.W_o = nn.Linear(self.d_model, self.d_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X [B, seq_len, d_model]
        B, seq_len, _ = X.shape 
        out_dim = (B, seq_len, self.num_heads, self.d_k)
        Q = torch.reshape(self.W_q(X), out_dim).transpose(2, 1)
        K = torch.reshape(self.W_k(X), out_dim).transpose(2, 1)
        V = torch.reshape(self.W_v(X), out_dim).transpose(2, 1)
        
        # Q, K, V are of shape [B, n_heads, seq_len, d_k]
        A = ((Q @ K.mT) / (self.d_k ** 0.5))

        if self.use_mask:
            mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=A.device),
                diagonal=1
            )
            A = A + mask

        A = self.dropout(softmax(A, dim=-1)) @ V

        # A is of shape [B, n_heads, seq_len, d_k]
        A = A.transpose(1, 2) # [B, seq_len, n_heads, d_k]
        A = torch.reshape(A, (B, seq_len, self.d_model)) # [B, seq_len, d_model]

        return self.dropout(self.W_o(A))

        