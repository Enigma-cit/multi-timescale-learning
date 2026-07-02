import torch
import torch.nn as nn


class ContinuumMemory(nn.Module): 
    """
    Continuum memory block with short-, medium-, and long-timescale modules.

    Each submodule is a simple feed-forward block; the scheduler decides which
    modules receive gradient / state updates on a given step.

    Forward always uses all memories; update frequencies modulate parameter
    updates (via masks) and/or internal state updates.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_blocks: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            ) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks:
            out = out + block(out)
        return out

    def timescale_indices(self):
        # Assume 0 = short, 1 = medium, 2 = long for num_blocks=3
        return {"short": 0, "medium": 1, "long": 2}
