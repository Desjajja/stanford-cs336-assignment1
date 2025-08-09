import torch
from torch import Tensor
from jaxtyping import Float
from einops import einsum
import math

def softmax(x: torch.Tensor, dim=-1) -> torch.Tensor:
    max_x = torch.amax(x, dim, keepdim=True)
    exp_x = torch.exp(x - max_x) # centered x
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: Float[Tensor, " batch ... queries d_k"],
    K: Float[Tensor, " batch ... keys d_k"],
    V: Float[Tensor, " batch ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " batch ... d_v"]:
    d_k = K.shape[-1]
    QK = einsum(
        Q, K,
        "... queries d_k, ... keys d_k -> ... queries keys"
    )
    if mask is not None:
        mask = torch.where(mask == 0, torch.tensor(-torch.inf), 0)
        QK += mask
    QK_proj = softmax(QK / math.sqrt(d_k))
    return einsum(
        QK_proj, V,
        "... queries seq_len, ... seq_len d_v -> ... queries d_v"
    )