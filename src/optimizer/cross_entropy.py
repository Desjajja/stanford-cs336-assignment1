import torch
from src.modules import functional as F
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(logits: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]):
    # PyTorch's logsumexp is numerically stable and implements the trick
    # of subtracting the max logit internally.
    # log_sum_exp has shape (batch_size,)
    log_sum_exp = torch.logsumexp(logits, dim=1)

    # Get the logits corresponding to the target indices.
    # We use gather() to select the correct logit from each row.
    # logit_targets has shape (batch_size,)
    logit_target = logits.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

    # Calculate the loss for each sample using the formula:
    # -logit[target] + log(sum(exp(logits)))
    loss = -logit_target + log_sum_exp

    # Return the average loss across the batch.
    return loss.mean()
