
import torch
import torch.nn as nn
from einops import rearrange, einsum


class RotaryPositionalEmbedding(nn.Module):
    inv_freq: torch.Tensor
    inv_freq_cache: torch.Tensor
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        super().__init__()
        self.dim = d_k
        self.base = theta
        self.max_seq_len = max_seq_len
        # Create the positional encodings: theta_i = base^(-2i/dim) = 1/base^(2i/dim)
        # shape: [dim // 2]
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
        )
        inv_freq_cache = einsum(
            torch.arange(max_seq_len), inv_freq,
            "i, d_k -> i d_k"
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("inv_freq_cache", inv_freq_cache, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k); token_position: (d_k)
        assert x.shape[-1] == self.dim
        # shape: [Batch, Attn_Head, Seq_Len, dim//2, 2]
        hidden_states_pairs = rearrange(
            x, "... (d_k_h pair) -> ... d_k_h pair", pair=2 # k_d_h: half length of k_d
        )
        # shape: [Batch, Attn_Head, Seq_Len, dim//2]
        hidden_states_complex = torch.view_as_complex(hidden_states_pairs)

        max_token_pos = int(token_positions.max())
        if max_token_pos < self.max_seq_len: # use cache
            freqs = self.inv_freq_cache[token_positions, ...]
        else:
            freqs = einsum(token_positions, self.inv_freq, "... seq_len, ... d_k_h -> ... seq_len d_k_h").float()
        freqs_cis = torch.polar(
            abs=torch.ones_like(freqs),
            angle=freqs
        )
        # shape: [Batch, Attn_Head, Seq_Len, dim//2]
        position_embed_complex = (
            torch.einsum(
                "...i,...i -> ...i",
                hidden_states_complex,
                freqs_cis
            )
        )
        # shape: [Batch, Attn_Head, Seq_Len, dim]
        position_embed = rearrange(torch.view_as_real(position_embed_complex), "... d_k_h pair -> ... (d_k_h pair)")
        return position_embed