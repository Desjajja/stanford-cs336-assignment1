import torch
import torch.nn as nn
from einops import einsum


class Swiglu(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            d_ff = ((d_ff % 64) + 1) * 64
        self.w1 = nn.Parameter(torch.randn(d_ff, d_model))
        self.w2 = nn.Parameter(torch.randn(d_model, d_ff))
        self.w3 = nn.Parameter(torch.randn(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj1 = einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff")
        x_proj2 = einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu = x_proj1 * torch.sigmoid(x_proj1)
        
        return einsum(
            self.w2, silu * x_proj2,
            "d_model d_ff, ... d_ff -> ... d_model"
        )
