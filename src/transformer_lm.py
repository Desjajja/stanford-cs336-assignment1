import torch.nn as nn
from collections import OrderedDict

from src.modules import Embedding, TransformerBlock, RMSNorm, Linear


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ) -> None:
        super().__init__()

        self.sublayers = nn.Sequential(
            OrderedDict(
                [
                    ("token_embeddings", Embedding(vocab_size, d_model)),
                    (
                        "layers",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (f"{i}", TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta))
                                    for i in range(num_layers)
                                ]
                            )
                        ),
                    ),
                    ("ln_final", RMSNorm(d_model)),
                    ("lm_head", Linear(d_model, vocab_size)),
                ]
            )
        )

    def forward(self, x):
        out = self.sublayers(x)
        return out
