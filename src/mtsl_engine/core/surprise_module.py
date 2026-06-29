import torch


def prediction_surprise(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Surprise based on prediction error.

    Use e.g. ||pred - target|| or KL divergence for probabilistic outputs.
    """
    return torch.norm(pred - target)