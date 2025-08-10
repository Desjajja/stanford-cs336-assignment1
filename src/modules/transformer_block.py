import torch
import torch.nn as nn
from src.modules import Swiglu, MultiheadSelfAttention, RotaryPositionalEmbedding as RoPE, Linear, RMSNorm
from jaxtyping import Float
from torch import Tensor


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        self.attn = MultiheadSelfAttention(d_model, num_heads)
        self.rope = RoPE(theta, d_model // num_heads, max_seq_len)
        self.ln1 = RMSNorm(d_model)
        self.ffn = Swiglu(d_model, d_ff)
        self.ln2 = RMSNorm(d_model)

    def forward(
        self, x: Float[Tensor, " ... sequence_length d_model"]
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        attn_out = self.attn(self.ln1(x), token_positions=torch.arange(x.shape[-2]), rope=self.rope)
        x = x + attn_out
        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out
        return x
