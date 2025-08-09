import torch
import torch.nn as nn
from einops import einsum, reduce

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        super().__init__()
        self.weights = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rms = torch.sqrt(
        #     torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps
        # ) # [... d_model] -> [... 1]
        
        rms = torch.sqrt(
            reduce(torch.square(x), "... d_model -> ... 1", 'mean') + self.eps
        )
        
        return einsum(
            x / rms, self.weights,
            "... d_model, d_model -> ... d_model"
        )