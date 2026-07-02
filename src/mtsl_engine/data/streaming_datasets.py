import torch
import numpy as np
 

class RotatingGaussiansStream:
    """
    Simple streaming data task:

    - Input x_t ∈ ℝ^d from a Gaussian whose mean rotates slowly on a circle.
    - Label y_t is a binary class depending on the quadrant of the mean.

    Used to test sample-efficiency and stability under distribution drift.
    """

    def __init__(self, dim: int = 2, radius: float = 3.0, drift_rate: float = 1e-3, seed: int | None = None):
        self.dim = dim
        self.radius = radius
        self.drift_rate = drift_rate
        self.theta = 0.0
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size: int):
        self.theta += self.drift_rate * batch_size
        mu = np.array([
            self.radius * np.cos(self.theta),
            self.radius * np.sin(self.theta),
        ])
        if self.dim > 2:
            mu = np.concatenate([mu, np.zeros(self.dim - 2)])

        x = self.rng.normal(mu, 1.0, size=(batch_size, self.dim))
        x_torch = torch.from_numpy(x).float()

        # Quadrant-based label in 2D
        y = ((mu[0] >= 0.0) ^ (mu[1] >= 0.0)).astype(np.int64)
        y_batch = torch.full((batch_size,), int(y), dtype=torch.long)
        return x_torch, y_batch
