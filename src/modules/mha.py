import torch
import torch.nn as nn
from src.modules import functional as F, Linear
from jaxtyping import Float, Int
from torch import Tensor
from einops import rearrange
from src.modules import RotaryPositionalEmbedding as RoPE
# from modules import RotaryPositionalEmbedding


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        # self.d_k = self.d_v = int(d_model / self.num_heads)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.q_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)
        # self.rope = RotaryPositionalEmbedding()

    def forward(
        self,
        x: Float[Tensor, " batch ... seq_len d_in"],
        token_positions: Int[Tensor, " seq_len"] | None = None,
        rope: RoPE | None = None,
    ) -> Float[Tensor, " batch ... seq_len d_out"]:
        seq = x.shape[-2]
        Q = rearrange(self.q_proj(x), "b seq (h d_k) -> b h seq d_k", h=self.num_heads)
        K = rearrange(self.k_proj(x), "b seq (h d_k) -> b h seq d_k", h=self.num_heads)
        V = rearrange(self.v_proj(x), "b seq (h d_v) -> b h seq d_v", h=self.num_heads)
        if token_positions is not None and rope is not None:
            Q = rope(Q, token_positions)
            K = rope(K, token_positions)
        attn_output = F.scaled_dot_product_attention(Q, K, V, mask=torch.tril(torch.ones(seq, seq)))
        # merge features across attention heads
        attn_output = self.out_proj(rearrange(attn_output, "b h seq e -> b seq (h e)"))
        return attn_output
