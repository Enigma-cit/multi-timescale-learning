import torch
import torch.nn as nn


class OptimizerMemory(nn.Module):
    """
    Optimizer-as-memory module. 

    Maintains a learned momentum-like state m_t in the same flattened parameter
    space as the model parameters. The module predicts a gradient trend g_hat_t
    and updates based on surprise ||g_t - g_hat_t||.
    """

    def __init__(self, param_dim: int, hidden_dim: int, surprise_threshold: float):
        super().__init__()
        self.param_dim = int(param_dim)
        self.surprise_threshold = float(surprise_threshold)

        self.trend_predictor = nn.Sequential(
            nn.Linear(self.param_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.param_dim),
        )

        self.register_buffer("momentum", torch.zeros(self.param_dim))

    def reset_state(self, device: torch.device | None = None) -> None:
        if device is None:
            self.momentum = torch.zeros(self.param_dim, device=self.momentum.device)
        else:
            self.momentum = torch.zeros(self.param_dim, device=device)

    def step(self, grad_vec: torch.Tensor) -> torch.Tensor:
        """
        Compute update direction given current gradient vector.
        """
        if grad_vec.dim() != 1:
            raise ValueError(f"grad_vec must be 1D, got shape {tuple(grad_vec.shape)}")

        if grad_vec.numel() != self.param_dim:
            raise ValueError(
                f"Gradient dimension mismatch: expected {self.param_dim}, got {grad_vec.numel()}"
            )

        if self.momentum.device != grad_vec.device:
            self.momentum = self.momentum.to(grad_vec.device)

        g_hat = self.trend_predictor(self.momentum)
        surprise = torch.norm(grad_vec - g_hat, p=2)

        alpha = torch.clamp(
            surprise / (self.surprise_threshold + 1e-8),
            min=0.0,
            max=1.0,
        )

        self.momentum = (1.0 - alpha) * self.momentum + alpha * grad_vec.detach()
        update = 0.5 * g_hat + 0.5 * grad_vec
        return update
