import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    def __init__(self, n_features, out_features, device=None, dtype=None) -> None:
        """
        linear transformation module. This function should accept the following parameters:
        ---
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        sigma = float(math.sqrt(2 / (n_features + out_features)))
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(out_features, n_features, device=device, dtype=dtype), std=sigma, a=-3 * sigma, b=3 * sigma
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.T)
