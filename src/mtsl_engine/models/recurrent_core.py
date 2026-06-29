import torch
import torch.nn as nn


class RecurrentCore(nn.Module):
    """
    Lightweight recurrent core (GRU-style) used as fast-timescale memory.

    The hidden state h_t acts as a short-term memory that can be coupled to
    continuum memories and optimizer memory.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        """
        x: [B, T, input_dim]
        h: [1, B, hidden_dim] or None
        """
        out, h_next = self.gru(x, h)
        return out, h_next