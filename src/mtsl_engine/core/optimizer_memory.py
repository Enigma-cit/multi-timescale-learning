import torch
import torch.nn as nn


class OptimizerMemory(nn.Module):
    """
    Optimizer-as-memory module.

    - Maintains a learned momentum state m_t.
    - Predicts gradient trend g_hat_t from past states.
    - Updates memory based on surprise: ||g_t - g_hat_t||.

    Designed to plug into training loops in place of a fixed optimizer.
    """

    def __init__(self, param_dim: int, hidden_dim: int, surprise_threshold: float):
        super().__init__()
        self.surprise_threshold = surprise_threshold

        self.trend_predictor = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, param_dim),
        )

        self.register_buffer("momentum", torch.zeros(param_dim))

    def step(self, grad_vec: torch.Tensor) -> torch.Tensor:
        """
        Compute update direction given current gradient vector.

        Returns:
          update direction Δθ_t (same shape as grad_vec)
        """
        g_hat = self.trend_predictor(self.momentum)
        surprise = torch.norm(grad_vec - g_hat)

        # Surprise-based gating: high surprise -> rely more on raw grad
        alpha = torch.clamp(surprise / (self.surprise_threshold + 1e-8), 0.0, 1.0)

        # Update momentum as a convex combination
        self.momentum = (1 - alpha) * self.momentum + alpha * grad_vec.detach()

        # Final update direction: combination of predicted trend and actual gradient
        update = 0.5 * g_hat + 0.5 * grad_vec
        return update