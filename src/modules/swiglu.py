import torch
import torch.nn as nn
from src.modules import Linear
from einops import einsum


class Swiglu(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            d_ff = ((d_ff % 64) + 1) * 64
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj1 = self.w1(x)
        x_proj2 = self.w3(x)
        silu = x_proj1 * torch.sigmoid(x_proj1)
        
        return self.w2(silu * x_proj2)
            # "d_model d_ff, ... d_ff -> ... d_model"
